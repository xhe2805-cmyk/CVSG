"""CVSG: Coalition Value and Synergy-Guided Multimodal Fusion.

The network predicts continuous sentiment from word-aligned text, audio and
visual sequences. It combines distributional modality encoding with learned
contribution-based allocation and pairwise interaction corrections.

Organization
------------
CVSGTextEncoder: contextual text representations followed by convolution.
CoalitionValueSynergyFusion: feature-conditioned modality values, pair
    proposals, bounded interaction gains, and training-time subset targets.
CVSGFusion: unimodal and fused Gaussian parameterization, sampling,
    regularization, and joint task supervision.
CVSGModel: the application-facing network coupling language, acoustic and
    visual feature extraction to the Gaussian fusion module.

Training versus inference
-------------------------
The seven nonempty coalitions are evaluated only in the supervised branch.
Their losses produce detached utility and pair-gain targets during training.
At inference, the fusion path uses its learned predictors without labels,
subset enumeration or coalition losses.

Compatibility
-------------
This documented release retains the existing layer names, tensor operations,
sampling sites, loss coefficients, defaults, and forward-method interfaces.
This is a documentation expansion, not a modification of the algorithm.
"""
import logging
import math
import torch
from torch import nn
from torch.nn import functional as F
from transformers.models.bert.modeling_bert import BertPreTrainedModel
from transformers.models.bert.modeling_bert import BertEmbeddings, BertEncoder
from transformers.activations import gelu, gelu_new
from transformers.activations import ACT2FN as _ACT2FN
from global_configs import TEXT_DIM, ACOUSTIC_DIM, VISUAL_DIM, DEVICE
from modules.transformer import TransformerEncoder

logger = logging.getLogger(__name__)
MODEL_ID = 'CVSG'
MODEL_NAME = 'Coalition Value and Synergy-Guided Multimodal Fusion'

ACT2FN = {'gelu': gelu, 'relu': F.relu, 'swish': _ACT2FN['swish'], 'gelu_new': gelu_new}

def _mask_pool(x,mask):
    """Compute a masked temporal mean without introducing trainable parameters.

    Args:
        x: Token representations of shape [batch, time, feature_width].
        mask: Valid-position indicators of shape [batch, time], or None.

    Returns:
        Per-utterance feature means of shape [batch, feature_width].

    Mask handling:
        The denominator is clamped to at least one valid position. A missing
        mask falls back to an ordinary mean along the temporal dimension.
        Padding values are excluded by zeroing their multiplicative weights.
    """
    if mask is None: return x.mean(1)
    w=mask[:, :x.size(1)].to(x.dtype).unsqueeze(-1)
    return (x*w).sum(1)/w.sum(1).clamp_min(1.)

def _av_mask(mask):
    """Build the temporal validity mask for audio and visual features.

    The language mask includes special tokens; the nonverbal features aligned
    with those tokens should normally be excluded from pooling and KL terms.
    For each sample, positions strictly between the first and last valid token
    are retained. A short sequence with at most two valid tokens falls back
    to its first position so downstream masked means remain defined.

    Args:
        mask: Language valid-token mask [batch, time].
    Returns:
        Boolean nonverbal validity mask [batch, time].
    """
    n=mask.long().sum(-1,keepdim=True)
    s=torch.arange(mask.size(1),device=mask.device).unsqueeze(0)
    m=(s>0)&(s<(n-1))
    return m | ((n<=2)&(s==0))

class CVSGTextEncoder(BertPreTrainedModel):
    """Encode the language stream with contextual BERT features and convolution.

    This component follows the same forward signature expected by the model's
    pretrained-language loading path. Text is first converted into contextual
    embeddings. A temporal convolution maps the BERT hidden width to the
    shared multimodal width used by the audio and visual encoders.

    The model returns both a terminal temporal feature and the full projected
    sequence. CVSG uses the projected sequence for Gaussian token encoding.
    The terminal feature remains available for interface compatibility.

    The encoder owns its embedding and Transformer parameters so that the
    training script can assign it a separate learning rate if desired.
    """
    def __init__(self, config, multimodal_config, d_l):
        """Initialize language embeddings, Transformer layers, and temporal projection.

        Args:
            config: Hugging Face configuration with attention and embedding sizes.
            multimodal_config: Retained constructor argument for model integration.
            d_l: Shared latent width after the temporal convolution.

        The convolution keeps sequence length constant with kernel size three,
        stride one, and padding one. Standard module initialization is preserved.
        """
        super().__init__(config)
        self.config = config

        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)

        self.d_l = d_l
        self.proj_l = nn.Conv1d(TEXT_DIM, self.d_l, kernel_size=3, stride=1, padding=1, bias=False)

        self.init_weights()

    def get_input_embeddings(self):
        """Expose the token embedding matrix to pretrained-model utilities.

        The returned embedding layer is the same object used during forward.
        This method provides the standard integration point for resizing or
        inspecting a vocabulary's learned input representations.
        """
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        """Replace the token embedding layer without changing the encoder interface.

        Args:
            value: Replacement torch.nn.Embedding-compatible module.

        This setter does not touch the temporal projection or attention layers.
        """
        self.embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune):
        """Remove selected attention heads from language encoder layers.

        Args:
            heads_to_prune: Mapping from layer index to head indices.

        Pruning is implemented by the existing attention-layer method. Ordinary
        training does not call this method; it remains an optional model utility.
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    def forward(
        self,
        input_ids,
                 
                   
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=None,
        output_hidden_states=None,
    ):
        """Produce projected language features from BERT token sequences.

        Args:
            input_ids: Token indices [batch, sequence_length].
            attention_mask: Optional valid-token indicators.
            token_type_ids: Optional sentence-segment indices.
            position_ids: Optional explicit positional indices.
            head_mask: Optional Transformer attention-head mask.
            inputs_embeds: Alternate embeddings; exclusive with input_ids.
            encoder_hidden_states: Optional cross-attention states for decoder mode.
            encoder_attention_mask: Optional mask for those decoder states.
            output_attentions: Optional attention reporting flag.
            output_hidden_states: Optional intermediate-state reporting flag.

        Returns:
            Tuple of a terminal projected feature and projected sequence output.
            The sequence is [batch, latent_width, sequence_length] to match the
            downstream CVSG multimodal feature interface.

        Operational detail:
            Attention masking and encoder flags follow the Hugging Face contract.
            The convolution is applied after contextualization and does not alter
            the length of the token sequence.
        """
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time"
            )
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError(
                "You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)
        if token_type_ids is None:
            token_type_ids = torch.zeros(
                input_shape, dtype=torch.long, device=device)

                                                                                                         
                                                                                     
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(
            attention_mask, input_shape, device
        )

                                                                          
                                                                                         
        if self.config.is_decoder and encoder_hidden_states is not None:
            (
                encoder_batch_size,
                encoder_sequence_length,
                _,
            ) = encoder_hidden_states.size()
            encoder_hidden_shape = (
                encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(
                    encoder_hidden_shape, device=device)
            encoder_extended_attention_mask = self.invert_attention_mask(
                encoder_attention_mask
            )
        else:
            encoder_extended_attention_mask = None

                                     
                                                    
                                                         
                                                                                  
                                                                                                               
        head_mask = self.get_head_mask(
            head_mask, self.config.num_hidden_layers)

        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
        )

       
        fused_embedding = embedding_output

        encoder_outputs = self.encoder(
            fused_embedding,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
        sequence_output = encoder_outputs[0]                             
                                     
                                                     

        outputs = sequence_output.transpose(1, 2)                                     
        outputs = self.proj_l(outputs)                                      
                                              
        pooled_output = outputs[:, :, -1]

                                     
                                                                        
                
                                                                
                                                                       
        return pooled_output, outputs






class CoalitionValueSynergyFusion(nn.Module):
    """Predict modality utility and pairwise gains from input-dependent features.

    The full inference path first constructs projected modality content.
    Value predictors condition on content, global context, and scale proxies.
    Their scores determine a smoothed softmax allocation across three sources.
    Pair proposal networks produce signed corrections from source content,
    while pair-gain networks estimate bounded interaction-control scores.

    During supervised training, seven auxiliary subset predictions are formed.
    Their negative smooth-L1 losses define a coalition-quality function.
    Three-player Shapley differences create value targets, and second-order
    subset differences create targets for the pair-gain networks.

    These quality-derived targets are detached from the prediction graph.
    They are supervisory signals, not features consumed at inference time.
    The pairwise gates are nonnegative; the signed correction direction is
    determined by each learned pair proposal, not by the gate itself.
    """
    PAIRS=((0,1),(0,2),(1,2))
                                                                        
    COALITIONS=((1,0,0),(0,1,0),(0,0,1),
                (1,1,0),(1,0,1),(0,1,1),(1,1,1))

    def __init__(self,dim):
        """Construct all predictors used by content routing and pair corrections.

        Args:
            dim: Feature width shared by the three modality representations.

        The module preserves the exact three pair indices (text/audio, text/visual,
        audio/visual) and the enumeration of seven nonempty coalitions. Each pair
        owns an independent proposal and gain-predictor network. A single auxiliary
        head is shared across all subset predictions to place their quality scores
        on the same supervised loss scale.

        Gain-output biases start at zero. That initialization is important for
        retaining the original pair-gate behavior at the start of training.
        """
        super().__init__()
        self.dim=dim
        self.modality_proj=nn.ModuleList([
            nn.Sequential(nn.LayerNorm(dim),nn.Linear(dim,dim),nn.GELU())
            for _ in range(3)])
        self.pair_proj=nn.ModuleList([
            nn.Sequential(nn.LayerNorm(dim*3),nn.Linear(dim*3,2*dim),
                          nn.GELU(),nn.Linear(2*dim,dim),nn.Tanh())
            for _ in range(3)])
        self.pair_gain=nn.ModuleList([
            nn.Sequential(nn.LayerNorm(dim*3+2),nn.Linear(dim*3+2,dim),
                          nn.GELU(),nn.Linear(dim,1))
            for _ in range(3)])
        self.modality_value=nn.ModuleList([
            nn.Sequential(nn.LayerNorm(2*dim+1),nn.Linear(2*dim+1,dim),
                          nn.GELU(),nn.Linear(dim,1))
            for _ in range(3)])
        self.coalition_head=nn.Sequential(nn.LayerNorm(dim),nn.Linear(dim,1))
        self.coalition_norm=nn.LayerNorm(dim)
        self.pair_out=nn.Sequential(nn.LayerNorm(dim),nn.Linear(dim,dim),
                                    nn.GELU(),nn.Linear(dim,dim))
        self.fused_norm=nn.LayerNorm(dim)
        for gain in self.pair_gain:
            nn.init.zeros_(gain[-1].bias)

    @staticmethod
    def shapley_targets(quality):
        """Compute exact marginal-value targets for the three-modality coalition game.

        Args:
            quality: Dictionary of batchwise supervised quality tensors keyed by
                integer subset bit masks. Bit 0 denotes text, bit 1 audio, and bit
                2 vision. Key zero is the prediction-zero reference quality.

        Returns:
            Tensor of three marginal contributions for each sample, ordered by
            text, audio, and vision. The trailing singleton dimension is retained.

        Computation:
            Every modality joins all four coalitions formed by the other two.
            Contributions use coefficients 1/3, 1/6, 1/6, and 1/3 according to
            coalition size. Targets have no gradient path to subset predictions,
            because their quality entries are detached before aggregation.
        """
        values=[]
        factorial={0:1/3,1:1/6,2:1/3}
        for idx in range(3):
            other=[j for j in range(3) if j!=idx]
            contrib=0.
            for sub in range(4):
                bits=sum((1<<other[t]) for t in range(2) if (sub>>t)&1)
                contrib=contrib+factorial[sub.bit_count()] * (quality[bits|(1<<idx)]-quality[bits])
            values.append(contrib)
        return torch.stack(values,dim=1)

    def forward(self, pooled, logvar, labels=None, use_value=True,
                use_synergy=True, use_uncertainty=True,
                use_coalition=True, synergy_supervision=True,
                utility_temp=1.5, pair_strength=.65):
        """Fuse three pooled Gaussian features and optionally supervise their utility.

        Args:
            pooled: Sequence of text, audio, and visual tensors [batch, dim].
            logvar: Batchwise [batch, 3] log-scale descriptors.
            labels: Optional continuous sentiment targets, used only for training.
            use_value: Whether learned value scores determine allocation weights.
            use_synergy: Whether pair-proposal corrections are added to content.
            use_uncertainty: Whether relative scale enters contribution predictors.
            use_coalition: Whether to evaluate supervised subset quality targets.
            synergy_supervision: Whether gain predictors fit pairwise targets.
            utility_temp: Temperature of the value-score softmax allocation.
            pair_strength: Residual scale on the output pair-correction pathway.

        Returns:
            full: Fused utterance-level representation [batch, dim].
            reg: Either scalar zero or coalition, value, and pair-target losses.
            diagnostic: Detached routing, utility scores, and synergy gain scores.

        Full-view content:
            A learned modality projection first produces comparable content vectors.
            A global mean context is given to every utility predictor. Centered,
            clipped negative log-variance values supply relative scale features.
            Softmax routing is mixed with uniform weights to avoid fully ignoring
            any active modality; the original mixing coefficient is 0.2.

        Pair correction:
            Each proposal receives two modality vectors and their product. Its
            gain predictor additionally receives the pair's scale descriptors.
            Pairwise contributions are gated by sigmoid(gain) and the geometric
            mean of the learned modality allocations. A learned output mapping,
            fixed residual scaling, and final layer normalization complete fusion.

        Coalition supervision:
            Subsets use scale-conditioned weighting and only their included pairs.
            The same auxiliary head predicts each subset's continuous sentiment.
            Negative samplewise smooth-L1 losses define detached coalition values.
            Shapley utility targets and pair-gain differences supervise their
            respective prediction networks via smooth-L1 losses.

        Evaluation:
            Passing labels=None bypasses coalition enumeration entirely. The
            learned allocation and pair-correction pathways remain active.
        """
        B=pooled[0].size(0)
        device,dtype=pooled[0].device,pooled[0].dtype
        h=[p(e) for p,e in zip(self.modality_proj,pooled)]
        context=torch.stack(h,dim=1).mean(dim=1)
        if not use_uncertainty:
            logvar=logvar*0.
                                                                               
                                                                                    
        precision_proxy=(-logvar).clamp(-6.,6.)
        precision_proxy=precision_proxy-precision_proxy.mean(dim=1,keepdim=True)
        pair_emb=[]; pair_value=[]
        for k,(i,j) in enumerate(self.PAIRS):
            x,y=h[i],h[j]
            proposal=self.pair_proj[k](torch.cat([x,y,x*y],dim=-1))
            score=self.pair_gain[k](torch.cat([x,y,x*y,
                            precision_proxy[:,i:i+1],precision_proxy[:,j:j+1]],dim=-1))
            pair_emb.append(proposal)
            pair_value.append(2.*torch.tanh(score))
        pair_value=torch.cat(pair_value,dim=1)
                                                                          
        value_pred=torch.cat([
            module(torch.cat([h[i],context,precision_proxy[:,i:i+1]],dim=-1))
            for i,module in enumerate(self.modality_value)],dim=1)
        if use_value:
            routing=torch.softmax(value_pred/utility_temp,dim=1)
            routing=0.8*routing+0.2/3.                                    
        else:
            routing=torch.full_like(value_pred,1./3.)
        base=sum(routing[:,i:i+1]*h[i] for i in range(3))
        if use_synergy:
            correction=base.new_zeros(base.shape)
            for k,(i,j) in enumerate(self.PAIRS):
                strength=torch.sqrt(routing[:,i:i+1]*routing[:,j:j+1])
                gate=torch.sigmoid(pair_value[:,k:k+1])
                correction=correction+strength*gate*pair_emb[k]
            full=self.fused_norm(base+pair_strength*self.pair_out(correction))
        else:
            full=self.fused_norm(base)
        reg=full.new_zeros(())
        diagnostic={'route':routing.detach(),'utility':value_pred.detach(),
                    'synergy':pair_value.detach()}
        if labels is not None and use_coalition:
            y=labels.reshape(-1,1)
                                                                    
                                                                                
            qualities={0:-F.smooth_l1_loss(torch.zeros_like(y),y,reduction='none').detach()}
            subset_losses=[]
            for subset in self.COALITIONS:
                code=sum((1<<i) for i,v in enumerate(subset) if v)
                co_mask=torch.tensor(subset,device=device,dtype=dtype).reshape(1,3)
                subset_logits=(precision_proxy/1.5).masked_fill(co_mask==0,-1e4)
                subset_w=torch.softmax(subset_logits,dim=1)
                subset_base=sum(subset_w[:,i:i+1]*h[i] for i in range(3))
                if use_synergy and sum(subset)>1:
                    sub_pair=subset_base.new_zeros(subset_base.shape)
                    for k,(i,j) in enumerate(self.PAIRS):
                        if subset[i] and subset[j]:
                            pair_gate=torch.sigmoid(pair_value[:,k:k+1])
                            sub_pair=sub_pair+torch.sqrt(subset_w[:,i:i+1]*subset_w[:,j:j+1])*pair_gate*pair_emb[k]
                    subset_base=subset_base+pair_strength*self.pair_out(sub_pair)
                subset_repr=self.coalition_norm(subset_base)
                subset_score=self.coalition_head(subset_repr)
                subset_losses.append(F.smooth_l1_loss(subset_score,y))
                qualities[code]=-F.smooth_l1_loss(subset_score.detach(),y,reduction='none')
            co_loss=torch.stack(subset_losses).mean()
                                                                                
                                                                                 
            target_value=self.shapley_targets(qualities).squeeze(-1).clamp(-3.,3.)
            val_loss=F.smooth_l1_loss(value_pred,target_value) if use_value else reg
            sy_loss=reg
            if use_synergy and synergy_supervision:
                gains=[]
                for i,j in self.PAIRS:
                    target=qualities[(1<<i)|(1<<j)]-qualities[1<<i]-qualities[1<<j]+qualities[0]
                    gains.append(target.squeeze(-1).clamp(-2.,2.))
                target_pairs=torch.stack(gains,dim=1)
                sy_loss=F.smooth_l1_loss(pair_value,target_pairs)
            reg=(co_loss,val_loss,sy_loss)
        return full,reg,diagnostic


class CVSGFusion(nn.Module):
    """Coordinate distributional encoding and supervised coalition-guided fusion.

    This module owns token-level Gaussian parameter networks for the three
    modalities and a separate Gaussian bottleneck for the fused representation.
    Mean pooling over valid temporal positions creates utterance descriptors,
    and averaged token variances generate one log-scale per modality.

    The coalition module supplies a full-view feature representation and, in
    training, the auxiliary losses for subset prediction, utility matching, and
    pair-gain matching. The final linear head regresses the sentiment score.
    A Gaussian prior KL loss regularizes token and fused posteriors.
    """
    def __init__(self,num_latents,dim):
        """Initialize four Gaussian parameterization networks and the fusion head.

        Args:
            num_latents: Retained constructor parameter for the model interface.
            dim: Common token, pooled, and fused latent feature width.

        The separate Gaussian heads share architectural dimensions but do not
        share trainable weights. Network names are retained to preserve checkpoint
        compatibility and the mapping of parameter tensors in state_dict().
        """
        super().__init__()
        self.d_l=dim
        self.encoder_l=nn.Sequential(nn.Linear(dim,1024),nn.ReLU())
        self.encoder_a=nn.Sequential(nn.Linear(dim,1024),nn.ReLU())
        self.encoder_v=nn.Sequential(nn.Linear(dim,1024),nn.ReLU())
        self.encoder=nn.Sequential(nn.Linear(dim,1024),nn.ReLU())
        self.fc_mu_l=nn.Linear(1024,dim);self.fc_std_l=nn.Linear(1024,dim)
        self.fc_mu_a=nn.Linear(1024,dim);self.fc_std_a=nn.Linear(1024,dim)
        self.fc_mu_v=nn.Linear(1024,dim);self.fc_std_v=nn.Linear(1024,dim)
        self.fc_mu=nn.Linear(1024,dim);self.fc_std=nn.Linear(1024,dim)
        self.fusion1=CoalitionValueSynergyFusion(dim)
        self.predictor=nn.Linear(dim,1)                                  
        self.configure_training_options()

    def reset_pair_gain_bias(self):
        """Restore zero initial pair-gain output biases after model initialization.

        The enclosing model initializes its trainable parameters and then invokes
        this method to enforce the intended neutral initial pair-gain prediction.
        It changes only biases of the last linear layer in each pair-gain network.
        """
        for mod in self.fusion1.pair_gain:
            nn.init.zeros_(mod[-1].bias)

    def configure_training_options(self,disable_coalition=False,disable_value=False,
                          disable_synergy=False,disable_uncertainty=False,
                          disable_synergy_supervision=False,kl_weight=.001,
                          coalition_weight=.08,value_weight=.04,
                          synergy_weight=.02,utility_temperature=1.5,
                          pair_strength=.65,deterministic_eval=True,clip_eval=True):
        """Set runtime and ablation switches without replacing any learned modules.

        Args:
            disable_coalition: Disable training-only supervised subset objectives.
            disable_value: Use uniform instead of learned modality allocation.
            disable_synergy: Remove pairwise correction from fusion.
            disable_uncertainty: Zero the input log-scale descriptors before fusion.
            disable_synergy_supervision: Omit pair-gain target matching.
            kl_weight: Coefficient of Gaussian prior regularization.
            coalition_weight: Weight of seven-subset prediction loss.
            value_weight: Weight of marginal-utility target matching.
            synergy_weight: Weight of pair-gain target matching.
            utility_temperature: Allocation-softmax temperature.
            pair_strength: Pair residual correction strength.
            deterministic_eval: Use posterior means in evaluation when enabled.
            clip_eval: Clamp evaluation predictions to the annotated score range.

        This method is intentionally parameter-free. It changes only Python-side
        control attributes so saved model tensor names remain stable across runs.
        """
        self.use_coalition=not bool(disable_coalition)
        self.use_value=not bool(disable_value)
        self.use_synergy=not bool(disable_synergy)
        self.use_uncertainty=not bool(disable_uncertainty)
        self.use_synergy_supervision=not bool(disable_synergy_supervision)
        self.kl_weight=float(kl_weight)
        self.coalition_weight=float(coalition_weight)
        self.value_weight=float(value_weight)
        self.synergy_weight=float(synergy_weight)
        self.utility_temperature=float(utility_temperature)
        self.pair_strength=float(pair_strength)
        self.deterministic_eval=bool(deterministic_eval)
        self.clip_eval=bool(clip_eval)

    def encode_l(self,x):
        """Parameterize language token Gaussian means and standard deviations.

        Args:
            x: Contextual language tokens with shape [batch, time, dim].
        Returns:
            Two tensors [batch, time, dim] for mean and positive scale.
        """
        h=self.encoder_l(x);return self.fc_mu_l(h),F.softplus(self.fc_std_l(h)-5.)+1e-5
    def encode_a(self,x):
        """Parameterize acoustic token Gaussian means and standard deviations.

        The acoustic encoder owns its own hidden projection and parameter heads.
        Its output shape matches that of the language token Gaussian parameters.
        """
        h=self.encoder_a(x);return self.fc_mu_a(h),F.softplus(self.fc_std_a(h)-5.)+1e-5
    def encode_v(self,x):
        """Parameterize visual token Gaussian means and standard deviations.

        Visual feature parameters are independent of the language and audio
        Gaussian heads, even though their output latent width is identical.
        """
        h=self.encoder_v(x);return self.fc_mu_v(h),F.softplus(self.fc_std_v(h)-5.)+1e-5
    def encode(self,x):
        """Parameterize the final Gaussian posterior over the fused representation.

        This is distinct from the three token-level Gaussian heads. It receives
        a single utterance-level feature and returns fused mean and scale vectors.
        """
        h=self.encoder(x);return self.fc_mu(h),F.softplus(self.fc_std(h)-5.)+1e-5
    @staticmethod
    def reparameterise(mu,std,stochastic=True):
        """Obtain a Gaussian latent sample or its posterior mean.

        Args:
            mu: Posterior mean tensor.
            std: Posterior standard deviation tensor, same shape as mu.
            stochastic: Draw a standard-normal tensor when True; otherwise use mu.
        Returns:
            Sampled or deterministic latent tensor without shape changes.

        The random draw location and sample count are preserved for reproducibility.
        """
        return mu+std*torch.randn_like(std) if stochastic else mu
    @staticmethod
    def _kl(mu,std,mask=None):
        """Compute diagonal-Gaussian KL divergence to a standard normal prior.

        Args:
            mu: Per-position means or utterance-level fused means.
            std: Matching positive standard deviations.
            mask: Optional validity mask for a temporal Gaussian representation.

        Returns:
            A scalar mean divergence, with invalid time positions excluded
            whenever a temporal mask is provided.

        The formula is one half of mu^2 + std^2 - 2 log(std) - 1.
        The denominator counts valid token-coordinate pairs, not padded entries.
        The fused representation uses the unmasked scalar mean form.
        """
        kl=.5*(mu.square()+std.square()-2*torch.log(std.clamp_min(1e-6))-1)
        if mask is None:return kl.mean()
        w=mask.to(kl.dtype).unsqueeze(-1)
        return (kl*w).sum()/(w.sum()*kl.size(-1)).clamp_min(1.)
    def forward(self,x_l,x_a,x_v,x_l_2D=None,x_a_2D=None,x_v_2D=None,
                label_ids=None,attention_mask=None):
        """Compute the full CVSG sentiment prediction and joint training loss.

        Args:
            x_l: Language features [batch, time, dim].
            x_a: Acoustic features [batch, time, dim].
            x_v: Visual features [batch, time, dim].
            x_l_2D, x_a_2D, x_v_2D: Retained unused interface arguments.
            label_ids: Optional continuous supervision for the current batch.
            attention_mask: Language validity mask [batch, time].

        Returns:
            pred: Continuous sentiment prediction [batch, 1].
            loss: Scalar task plus auxiliary regularization objective when labeled.

        Distributional encoding:
            Language uses the ordinary attention mask. Acoustic and visual
            streams exclude the first and terminal special-token positions.
            Token latents are sampled during training; in deterministic evaluation,
            posterior means replace Gaussian samples in both encoding stages.
            Masked utterance means provide the three inputs to coalition fusion.
            Token posterior variances generate relative scale descriptors.

        Full-view prediction:
            The fusion block performs utility-weighted content aggregation and
            optionally incorporates signed pairwise residual corrections. Its
            output enters an independent fused Gaussian bottleneck, followed by
            a linear sentiment regressor. Clipping is an evaluation-only option.

        Objective:
            Absolute, smooth-L1, and squared regression errors train prediction.
            During training, coalition supervision supplies the three auxiliary
            objectives. Gaussian prior KL terms regularize the unimodal and fused
            distributions. All existing coefficients and their order are retained.
        """
        if attention_mask is None:
            attention_mask=torch.ones(x_l.shape[:2],device=x_l.device,dtype=torch.bool)
        mask=attention_mask.bool()
        amask=_av_mask(mask)
        states=[self.encode_l(x_l),self.encode_a(x_a),self.encode_v(x_v)]
        stoch=self.training or not self.deterministic_eval
        z=[self.reparameterise(mu,std,stoch)*m.unsqueeze(-1)
            for (mu,std),m in zip(states,(mask,amask,amask))]
        pooled=[_mask_pool(z[0],mask),_mask_pool(z[1],amask),_mask_pool(z[2],amask)]
        logvar=torch.cat([
            torch.log(_mask_pool(std.square(),valid).mean(-1,keepdim=True).clamp_min(1e-5))
            for (_,std),valid in zip(states,(mask,amask,amask))],dim=1)
        fused,coalition_terms,_=self.fusion1(pooled,logvar,
            label_ids if (self.training and label_ids is not None) else None,
            use_value=self.use_value,use_synergy=self.use_synergy,
            use_uncertainty=self.use_uncertainty,
            use_coalition=self.use_coalition,
            synergy_supervision=self.use_synergy_supervision,
            utility_temp=self.utility_temperature,pair_strength=self.pair_strength)
        fused_mu,fused_std=self.encode(fused)
        zz=self.reparameterise(fused_mu,fused_std,stoch)
        pred=self.predictor(zz)
        if not self.training and self.clip_eval:
            pred=pred.clamp(-3.,3.)
        loss=pred.new_zeros(())
        if label_ids is not None:
            y=label_ids.reshape(-1);p=pred.reshape(-1)
            loss=F.l1_loss(p,y)+.15*F.smooth_l1_loss(p,y)+.03*F.mse_loss(p,y)
            if isinstance(coalition_terms,tuple):
                co,v,sy=coalition_terms
                loss=loss+self.coalition_weight*co
                if self.use_value:loss=loss+self.value_weight*v
                if self.use_synergy and self.use_synergy_supervision:
                    loss=loss+self.synergy_weight*sy
            if self.kl_weight:
                kl=sum(self._kl(mu,std,m) for (mu,std),m in zip(states,(mask,amask,amask)))
                loss=loss+self.kl_weight*(kl+self._kl(fused_mu,fused_std))
        return pred,loss

    def test(self,x_l,x_a,x_v,attention_mask=None):
        """Run the final inference path without coalition training targets.

        Args:
            x_l, x_a, x_v: Encoded modality sequences.
            attention_mask: Valid text positions, used to derive nonverbal masks.
        Returns:
            Batchwise sentiment prediction tensor without auxiliary losses.
        """
        return self.forward(x_l,x_a,x_v,None,None,None,None,attention_mask)[0]


class CVSGModel(BertPreTrainedModel):
    """End-to-end CVSG model for word-aligned multimodal sentiment regression.

    The language stream is contextualized using the pretrained text encoder.
    Acoustic and visual features receive independent temporal convolution and
    Transformer encoding. Each stream is mapped to the same latent width and
    passed into the Gaussian coalition-guided fusion network.

    During training, return_loss=True additionally computes task and auxiliary
    objectives. In standard inference, the model emits only predictions.
    """
    def __init__(self,config,multimodal_config):
        """Construct the language, acoustic, visual, and fusion submodules.

        Args:
            config: Pretrained language encoder configuration.
            multimodal_config: Retained multimodal settings container.

        Explicit gain-bias reset after pretrained-model initialization preserves
        the existing initial condition of the pairwise gain networks.
        """
        super().__init__(config)
        self.num_labels=config.num_labels
        self.d_l=50
        self.bert=CVSGTextEncoder(config,multimodal_config,self.d_l)
        self.attn_dropout=.5
        self.proj_a=nn.Conv1d(ACOUSTIC_DIM,self.d_l,3,padding=1,bias=False)
        self.proj_v=nn.Conv1d(VISUAL_DIM,self.d_l,3,padding=1,bias=False)
        self.transa=self.get_network(layers=3)
        self.transv=self.get_network(layers=3)
        self.fusion=CVSGFusion(0,self.d_l)
        self.init_weights()
        self.fusion.reset_pair_gain_bias()
    def get_network(self,self_type='l',layers=3):
        """Build a temporal Transformer block for acoustic or visual encoding.

        Args:
            self_type: Retained network-selection identifier.
            layers: Number of Transformer encoder layers.
        Returns:
            TransformerEncoder initialized with unchanged dropout settings.
        """
        return TransformerEncoder(embed_dim=self.d_l,num_heads=5,layers=layers,
                    attn_dropout=self.attn_dropout,relu_dropout=.3,res_dropout=.3,
                    embed_dropout=.2,attn_mask=False)
    def _get_features(self,input_ids,visual,acoustic,attention_mask,token_type_ids=None,
                    position_ids=None,head_mask=None,inputs_embeds=None,
                    output_attentions=None,output_hidden_states=None):
        """Extract temporally aligned modality features before Gaussian encoding.

        Args:
            input_ids: Batchwise token indices for language.
            visual: Word-aligned visual sequence features.
            acoustic: Word-aligned acoustic sequence features.
            attention_mask: Valid language token positions.
            token_type_ids: Optional language segment indices.
            position_ids: Optional language position indices.
            head_mask: Optional language attention-head mask.
            inputs_embeds: Optional direct language embeddings.
            output_attentions: Optional language attention reporting flag.
            output_hidden_states: Optional language intermediate-state flag.

        Returns:
            Tuple of aligned language, acoustic, and visual token embeddings,
            each shaped [batch, time, latent_width].
        """
        _,seq=self.bert(input_ids,attention_mask=attention_mask,token_type_ids=token_type_ids,
                        position_ids=position_ids,head_mask=head_mask,inputs_embeds=inputs_embeds,
                        output_attentions=output_attentions,output_hidden_states=output_hidden_states)
        l=seq.transpose(1,2)
        a=self.transa(self.proj_a(acoustic.transpose(1,2)).permute(2,0,1)).permute(1,0,2)
        v=self.transv(self.proj_v(visual.transpose(1,2)).permute(2,0,1)).permute(1,0,2)
        return l,a,v
    def forward(self,input_ids,visual,acoustic,label_ids=None,attention_mask=None,
                token_type_ids=None,position_ids=None,head_mask=None,inputs_embeds=None,
                labels=None,output_attentions=None,output_hidden_states=None,return_loss=False):
        """Execute end-to-end multimodal sentiment regression or supervised training.

        Args:
            input_ids: Text token identifiers.
            visual, acoustic: Aligned nonverbal feature tensors.
            label_ids: Training sentiment targets, if available.
            attention_mask: Text validity mask; defaults to all ones.
            token_type_ids, position_ids, head_mask: Optional text-encoder options.
            inputs_embeds: Optional externally supplied token embedding tensors.
            labels: Alternate name for training labels.
            output_attentions, output_hidden_states: Text encoder output options.
            return_loss: Return (predictions, loss) instead of predictions only.

        Returns:
            Sentiment predictions, or a prediction-and-loss tuple for training.

        No target labels are fed into the feature extraction path. Labels are
        passed exclusively to the supervised objective when return_loss is true.
        """
        if attention_mask is None: attention_mask=torch.ones_like(input_ids)
        l,a,v=self._get_features(input_ids,visual,acoustic,attention_mask,token_type_ids,
                position_ids,head_mask,inputs_embeds,output_attentions,output_hidden_states)
        y=label_ids if label_ids is not None else labels
        pred,loss=self.fusion(l,a,v,None,None,None,y if return_loss else None,attention_mask)
        return (pred,loss) if return_loss else pred
    def test(self,input_ids,visual,acoustic,attention_mask=None,token_type_ids=None,**kwargs):
        """Produce test-time predictions using learned full-view fusion parameters.

        The test interface matches the trainer's evaluation calls. Auxiliary
        coalition supervision is disabled because no training targets are passed.
        """
        return self.forward(input_ids,visual,acoustic,None,attention_mask,token_type_ids)

