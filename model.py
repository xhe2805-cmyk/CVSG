"""CVSG: Coalition Value and Synergy-Guided Multimodal Fusion.

Three modality-specific Gaussian encoders provide pooled representations and
posterior-scale descriptors. Coalition-based training targets supervise
feature-conditioned utility scores and pairwise interaction gains. The
inference path uses the learned utility and gain predictors without labels.
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
    if mask is None: return x.mean(1)
    w=mask[:, :x.size(1)].to(x.dtype).unsqueeze(-1)
    return (x*w).sum(1)/w.sum(1).clamp_min(1.)

def _av_mask(mask):
    n=mask.long().sum(-1,keepdim=True)
    s=torch.arange(mask.size(1),device=mask.device).unsqueeze(0)
    m=(s>0)&(s<(n-1))
    return m | ((n<=2)&(s==0))

class CVSGTextEncoder(BertPreTrainedModel):
    def __init__(self, config, multimodal_config, d_l):
        super().__init__(config)
        self.config = config

        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)

        self.d_l = d_l
        self.proj_l = nn.Conv1d(TEXT_DIM, self.d_l, kernel_size=3, stride=1, padding=1, bias=False)

        self.init_weights()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune):
        """ Prunes heads of the model.
            heads_to_prune: dict of {layer_num: list of heads to prune in this layer}
            See base class PreTrainedModel
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
        """Return projected text features and the sequence representation."""
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
    """Active fusion path: 7 train-only coalition heads + value-guided full fusion.

    Predictive coalition values are defined in supervised *loss* space; they are
    neither Shapley causal attributions nor mutual-information estimates.
    Labels are used ONLY to form detached training targets, never at inference.
    """
    PAIRS=((0,1),(0,2),(1,2))
                                                                        
    COALITIONS=((1,0,0),(0,1,0),(0,0,1),
                (1,1,0),(1,0,1),(0,1,1),(1,1,1))

    def __init__(self,dim):
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
        """Exact 3-player Shapley formula on DETACHED supervised loss quality.

        quality keys are 3-bit integer subset masks; 0 is neutral-score baseline.
        The target uses the training label but is never used as a forward input.
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
    """Gaussian encoding, coalition supervision, and full-view sentiment prediction."""
    def __init__(self,num_latents,dim):
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
        for mod in self.fusion1.pair_gain:
            nn.init.zeros_(mod[-1].bias)

    def configure_training_options(self,disable_coalition=False,disable_value=False,
                          disable_synergy=False,disable_uncertainty=False,
                          disable_synergy_supervision=False,kl_weight=.001,
                          coalition_weight=.08,value_weight=.04,
                          synergy_weight=.02,utility_temperature=1.5,
                          pair_strength=.65,deterministic_eval=True,clip_eval=True):
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
        h=self.encoder_l(x);return self.fc_mu_l(h),F.softplus(self.fc_std_l(h)-5.)+1e-5
    def encode_a(self,x):
        h=self.encoder_a(x);return self.fc_mu_a(h),F.softplus(self.fc_std_a(h)-5.)+1e-5
    def encode_v(self,x):
        h=self.encoder_v(x);return self.fc_mu_v(h),F.softplus(self.fc_std_v(h)-5.)+1e-5
    def encode(self,x):
        h=self.encoder(x);return self.fc_mu(h),F.softplus(self.fc_std(h)-5.)+1e-5
    @staticmethod
    def reparameterise(mu,std,stochastic=True):
        return mu+std*torch.randn_like(std) if stochastic else mu
    @staticmethod
    def _kl(mu,std,mask=None):
        kl=.5*(mu.square()+std.square()-2*torch.log(std.clamp_min(1e-6))-1)
        if mask is None:return kl.mean()
        w=mask.to(kl.dtype).unsqueeze(-1)
        return (kl*w).sum()/(w.sum()*kl.size(-1)).clamp_min(1.)
    def forward(self,x_l,x_a,x_v,x_l_2D=None,x_a_2D=None,x_v_2D=None,
                label_ids=None,attention_mask=None):
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
        return self.forward(x_l,x_a,x_v,None,None,None,None,attention_mask)[0]


class CVSGModel(BertPreTrainedModel):
    """Multimodal encoder and coalition-guided fusion network."""
    def __init__(self,config,multimodal_config):
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
        return TransformerEncoder(embed_dim=self.d_l,num_heads=5,layers=layers,
                    attn_dropout=self.attn_dropout,relu_dropout=.3,res_dropout=.3,
                    embed_dropout=.2,attn_mask=False)
    def _get_features(self,input_ids,visual,acoustic,attention_mask,token_type_ids=None,
                    position_ids=None,head_mask=None,inputs_embeds=None,
                    output_attentions=None,output_hidden_states=None):
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
        if attention_mask is None: attention_mask=torch.ones_like(input_ids)
        l,a,v=self._get_features(input_ids,visual,acoustic,attention_mask,token_type_ids,
                position_ids,head_mask,inputs_embeds,output_attentions,output_hidden_states)
        y=label_ids if label_ids is not None else labels
        pred,loss=self.fusion(l,a,v,None,None,None,y if return_loss else None,attention_mask)
        return (pred,loss) if return_loss else pred
    def test(self,input_ids,visual,acoustic,attention_mask=None,token_type_ids=None,**kwargs):
        return self.forward(input_ids,visual,acoustic,None,attention_mask,token_type_ids)

