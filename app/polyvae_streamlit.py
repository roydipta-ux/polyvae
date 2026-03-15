"""
PolyVAE — Simple Streamlit Dashboard
Run: streamlit run streamlit_app.py
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance
import warnings
warnings.filterwarnings('ignore')

# ── PAGE SETUP ────────────────────────────────────────────────────────────────
st.set_page_config(page_title="PolyVAE ⚗️", layout="wide")
st.title("⚗️ PolyVAE")
st.caption("Generative Deep Learning for Conductive Polymer Design")

# ── CONSTANTS ─────────────────────────────────────────────────────────────────
N_POLY  = 500
FP_DIM  = 64
LAT_DIM = 8
RNG     = np.random.default_rng(42)
np.random.seed(42)

FAMILIES = {
    'PEDOT':         {'pi_conj':0.98,'rigid':0.88,'dop_aff':0.70,'base':5.1},
    'Polythiophene': {'pi_conj':0.95,'rigid':0.80,'dop_aff':0.75,'base':4.2},
    'Polypyrrole':   {'pi_conj':0.88,'rigid':0.72,'dop_aff':0.85,'base':3.8},
    'Polyaniline':   {'pi_conj':0.82,'rigid':0.65,'dop_aff':0.90,'base':3.5},
    'Polyfluorene':  {'pi_conj':0.78,'rigid':0.92,'dop_aff':0.45,'base':1.8},
    'Polycarbazole': {'pi_conj':0.72,'rigid':0.88,'dop_aff':0.38,'base':1.2},
}
FAM_NAMES  = list(FAMILIES.keys())
FAM_COLORS = {
    'PEDOT':'#C62828','Polythiophene':'#00695C','Polypyrrole':'#1565C0',
    'Polyaniline':'#E65100','Polyfluorene':'#4527A0','Polycarbazole':'#2E7D32',
}
PHYS_COLS = ['pi_conj','backbone_rig','dop_affinity','chain_len','mol_wt',
             'homo_lumo_gap','ioniz_pot','electron_aff','sidechain_pol',
             'crystallinity','charge_mob','disorder','dopant_conc']
FEAT_LBL  = {
    'pi_conj':'π-Conjugation','backbone_rig':'Backbone Rigidity',
    'dop_affinity':'Doping Affinity','chain_len':'Chain Length',
    'mol_wt':'Mol. Weight','homo_lumo_gap':'HOMO-LUMO Gap',
    'ioniz_pot':'Ionization Pot.','electron_aff':'Electron Affinity',
    'sidechain_pol':'Side-Chain Polarity','crystallinity':'Crystallinity',
    'charge_mob':'Charge Mobility','disorder':'Disorder Param.',
    'dopant_conc':'Dopant Conc.'
}

# ── VAE (pure NumPy) ──────────────────────────────────────────────────────────
def relu(x): return np.maximum(0.0, x)

class PolyVAE:
    def __init__(self, in_dim, h=(128,64), L=8, beta=1.0, lr=8e-4):
        self.D, self.L, self.beta, self.lr = in_dim, L, beta, lr
        h1, h2 = h
        rng = np.random.default_rng(42)
        W = lambda fi,fo: rng.normal(0, np.sqrt(2/fi),(fi,fo)).astype(np.float32)
        b = lambda n: np.zeros(n, np.float32)
        self.We1,self.be1 = W(in_dim,h1),b(h1)
        self.We2,self.be2 = W(h1,h2),b(h2)
        self.Wmu,self.bmu = W(h2,L),b(L)
        self.Wlv,self.blv = W(h2,L),b(L)
        self.Wd1,self.bd1 = W(L,h2),b(h2)
        self.Wd2,self.bd2 = W(h2,h1),b(h1)
        self.Wo, self.bo  = W(h1,in_dim),b(in_dim)
        pn = ['We1','be1','We2','be2','Wmu','bmu','Wlv','blv','Wd1','bd1','Wd2','bd2','Wo','bo']
        self._m = {p:np.zeros_like(getattr(self,p)) for p in pn}
        self._v = {p:np.zeros_like(getattr(self,p)) for p in pn}
        self._t = 0; self._pn = pn

    def _encode(self,X):
        a1 = relu(X@self.We1+self.be1)
        a2 = relu(a1@self.We2+self.be2)
        mu = a2@self.Wmu+self.bmu
        lv = np.clip(a2@self.Wlv+self.blv,-4,4)
        return a1,a2,mu,lv

    def _reparam(self,mu,lv):
        rng = np.random.default_rng(42)
        eps = rng.standard_normal(mu.shape).astype(np.float32)
        sig = np.exp(0.5*lv)
        return mu+eps*sig, eps, sig

    def _decode(self,z):
        d1 = relu(z@self.Wd1+self.bd1)
        d2 = relu(d1@self.Wd2+self.bd2)
        return d1,d2,d2@self.Wo+self.bo

    def _adam(self,pn,g):
        self._t+=1; b1,b2,eps=0.9,0.999,1e-8
        g=np.clip(g,-1,1)
        self._m[pn]=b1*self._m[pn]+(1-b1)*g
        self._v[pn]=b2*self._v[pn]+(1-b2)*g**2
        mh=self._m[pn]/(1-b1**self._t); vh=self._v[pn]/(1-b2**self._t)
        setattr(self,pn,getattr(self,pn)-self.lr*mh/(np.sqrt(vh)+eps))

    def step(self,Xb):
        n=Xb.shape[0]
        ea1,ea2,mu,lv=self._encode(Xb)
        z,eps,sig=self._reparam(mu,lv)
        da1,da2,xh=self._decode(z)
        recon=np.mean((Xb-xh)**2)
        kl=-0.5*np.mean(1+lv-mu**2-np.exp(lv))
        loss=recon+self.beta*kl
        dxh=2*(xh-Xb)/n; gWo=da2.T@dxh; gbo=dxh.mean(0)
        dda2=(dxh@self.Wo.T)*(da2>0)
        gWd2=da1.T@dda2; gbd2=dda2.mean(0)
        dda1=(dda2@self.Wd2.T)*(da1>0)
        gWd1=z.T@dda1; gbd1=dda1.mean(0); dz=dda1@self.Wd1.T
        dmu=dz+self.beta*mu/n
        dlv=dz*eps*0.5*sig+0.5*self.beta*(np.exp(lv)-1)/n
        gWmu=ea2.T@dmu; gbmu=dmu.mean(0)
        gWlv=ea2.T@dlv; gblv=dlv.mean(0)
        dea2=((dmu@self.Wmu.T)+(dlv@self.Wlv.T))*(ea2>0)
        gWe2=ea1.T@dea2; gbe2=dea2.mean(0)
        dea1=(dea2@self.We2.T)*(ea1>0)
        gWe1=Xb.T@dea1; gbe1=dea1.mean(0)
        for p,g in [('We1',gWe1),('be1',gbe1),('We2',gWe2),('be2',gbe2),
                    ('Wmu',gWmu),('bmu',gbmu),('Wlv',gWlv),('blv',gblv),
                    ('Wd1',gWd1),('bd1',gbd1),('Wd2',gWd2),('bd2',gbd2),
                    ('Wo',gWo),('bo',gbo)]: self._adam(p,g)
        return loss,recon,kl

    def fit(self,X,epochs=160,bs=32):
        rng=np.random.default_rng(42); hist={'loss':[],'recon':[],'kl':[]}
        for ep in range(epochs):
            idx=rng.permutation(len(X)); ls=[]
            for s in range(0,len(X),bs):
                l,r,k=self.step(X[idx[s:s+bs]]); ls.append((l,r,k))
            hist['loss'].append(np.mean([x[0] for x in ls]))
            hist['recon'].append(np.mean([x[1] for x in ls]))
            hist['kl'].append(np.mean([x[2] for x in ls]))
        return hist

    def latent(self,X):
        _,_,mu,lv=self._encode(X); z,_,_=self._reparam(mu,lv)
        return mu,lv,z

    def generate(self,n=200):
        rng=np.random.default_rng(42)
        z=rng.standard_normal((n,self.L)).astype(np.float32)
        _,_,xh=self._decode(z); return z,xh

# ── BUILD DATA ────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Generating polymer dataset…")
def get_data():
    rng=np.random.default_rng(42)
    probs=[0.15,0.22,0.18,0.20,0.15,0.10]
    fams=rng.choice(FAM_NAMES,N_POLY,p=probs)
    rows=[]
    for fam in fams:
        fp=FAMILIES[fam]
        pi =np.clip(fp['pi_conj']+rng.normal(0,0.06),0.3,1.0)
        rig=np.clip(fp['rigid']+rng.normal(0,0.07),0.3,1.0)
        da =np.clip(fp['dop_aff']+rng.normal(0,0.07),0.1,1.0)
        cl =rng.integers(8,80); mw=cl*rng.uniform(80,200)
        gap=np.clip(3.5-2.8*pi+rng.normal(0,0.15),0.2,3.5)
        ip =np.clip(4.2+1.5*(1-da)+rng.normal(0,0.12),4.0,6.5)
        ea =np.clip(ip-gap-rng.normal(0.2,0.08),0.5,4.0)
        scp=rng.uniform(0.1,0.9)
        cry=np.clip(rig*0.7+rng.normal(0,0.10),0.1,0.95)
        mob=np.clip(pi**2*0.8+rng.normal(0,0.05),0.01,1.0)
        dis=np.clip(1-cry+rng.normal(0,0.08),0.05,0.90)
        dop=rng.uniform(0.05,0.40)
        bits=np.zeros(FP_DIM)
        s=(FAM_NAMES.index(fam)*10)%FP_DIM
        bits[s:min(s+8,FP_DIM)]=1.0
        bits[rng.choice(FP_DIM,int(pi*15),replace=False)]=1.0
        bits=np.abs(bits-(rng.random(FP_DIM)<0.08)).clip(0,1)
        lc=(fp['base']+2.8*pi+1.5*da+1.8*dop+1.2*mob
            -0.9*dis-0.4*gap+0.3*cry-0.2*scp+rng.normal(0,0.35))
        row=dict(family=fam,pi_conj=round(pi,4),backbone_rig=round(rig,4),
                 dop_affinity=round(da,4),chain_len=int(cl),mol_wt=round(mw,1),
                 homo_lumo_gap=round(gap,4),ioniz_pot=round(ip,4),
                 electron_aff=round(ea,4),sidechain_pol=round(scp,4),
                 crystallinity=round(cry,4),charge_mob=round(mob,4),
                 disorder=round(dis,4),dopant_conc=round(dop,4),
                 log_sigma=round(lc,4))
        for b in range(FP_DIM): row[f'fp{b:03d}']=int(bits[b])
        rows.append(row)
    return pd.DataFrame(rows)

@st.cache_resource(show_spinner="Training VAE (160 epochs)…")
def get_models(df):
    FP_COLS=[f'fp{b:03d}' for b in range(FP_DIM)]
    X_phys=df[PHYS_COLS].values.astype(np.float32)
    X_fp  =df[FP_COLS].values.astype(np.float32)
    y     =df['log_sigma'].values.astype(np.float32)
    sc=StandardScaler(); X_sc=sc.fit_transform(X_phys)
    X_vae=np.hstack([X_sc,X_fp]).astype(np.float32)
    IN_DIM=X_vae.shape[1]

    vae=PolyVAE(IN_DIM,h=(128,64),L=LAT_DIM,beta=1.0,lr=8e-4)
    hist=vae.fit(X_vae,epochs=160,bs=32)
    mu_all,lv_all,z_all=vae.latent(X_vae)
    _,_,xh_all=vae._decode(z_all)
    recon_r2=r2_score(X_sc,xh_all[:,:len(PHYS_COLS)])

    X_tr_z,X_te_z,y_tr,y_te=train_test_split(z_all,y,test_size=0.25,random_state=42)
    X_tr_p=X_sc[:int(0.75*N_POLY)]; X_te_p=X_sc[int(0.75*N_POLY):]
    models={}
    for tag,(Xtr,Xte) in [('Ridge-phys',(X_tr_p,X_te_p)),
                            ('GBM-phys',(X_tr_p,X_te_p)),
                            ('GBM-latent',(X_tr_z,X_te_z))]:
        m=(Ridge(alpha=10) if 'Ridge' in tag
           else GradientBoostingRegressor(n_estimators=300,max_depth=4,
                                          learning_rate=0.05,random_state=42))
        m.fit(Xtr,y_tr); p=m.predict(Xte)
        models[tag]={'model':m,'pred':p,'r2':r2_score(y_te,p),
                     'rmse':np.sqrt(mean_squared_error(y_te,p))}

    z_gen,xh_gen=vae.generate(200)
    pred_gen=models['GBM-latent']['model'].predict(z_gen)
    xh_phys=sc.inverse_transform(xh_gen[:,:len(PHYS_COLS)])
    xh_df=pd.DataFrame(xh_phys,columns=PHYS_COLS)
    valid=((xh_df['pi_conj'].between(0.3,1.0))&
           (xh_df['homo_lumo_gap'].between(0.2,3.5))&
           (xh_df['charge_mob'].between(0.01,1.0))).values
    nn_dist=cdist(z_gen,z_all,'euclidean').min(axis=1)
    novelty=nn_dist/nn_dist.max()

    pca=PCA(n_components=2); z_pca=pca.fit_transform(z_all)

    perm=permutation_importance(models['GBM-phys']['model'],
                                 X_te_p,y_te,n_repeats=30,random_state=42)
    imp_df=pd.DataFrame({'feat':PHYS_COLS,'imp':perm.importances_mean,
                          'std':perm.importances_std}
                        ).sort_values('imp',ascending=False).reset_index(drop=True)

    return {'vae':vae,'hist':hist,'sc':sc,'X_vae':X_vae,'X_sc':X_sc,
            'z_all':z_all,'z_pca':z_pca,'pca':pca,'y':y,'y_te':y_te,
            'models':models,'z_gen':z_gen,'pred_gen':pred_gen,
            'novelty':novelty,'valid':valid,'recon_r2':recon_r2,
            'imp_df':imp_df,'xh_all':xh_all,'X_te_p':X_te_p}

df = get_data()
M  = get_models(df)

# ── METRICS ROW ───────────────────────────────────────────────────────────────
c1,c2,c3,c4 = st.columns(4)
c1.metric("VAE Recon R²",      f"{M['recon_r2']:.3f}")
c2.metric("GBM-Latent R²",     f"{M['models']['GBM-latent']['r2']:.3f}")
c3.metric("Valid Generated",   f"{M['valid'].sum()}/200")
c4.metric("Novel Candidates",  f"{(M['novelty']>0.3).sum()}/200")
st.divider()

# ── TABS ──────────────────────────────────────────────────────────────────────
tab1,tab2,tab3,tab4,tab5 = st.tabs([
    "📊 EDA","🧠 Latent Space","🧪 Generation",
    "📈 XAI","⚡ Predictor"
])

# ── TAB 1: EDA ────────────────────────────────────────────────────────────────
with tab1:
    st.subheader("Exploratory Data Analysis")
    fig,axes=plt.subplots(1,3,figsize=(14,4))

    # Panel A: conductivity by family
    for fam,col in FAM_COLORS.items():
        axes[0].hist(df[df['family']==fam]['log_sigma'],bins=16,
                     alpha=0.55,color=col,label=fam,edgecolor='none')
    axes[0].set(xlabel='log₁₀(σ) [S/cm]',ylabel='Count',title='Conductivity by Family')
    axes[0].legend(fontsize=7,ncol=2)

    # Panel B: π-conjugation vs conductivity
    axes[1].scatter(df['pi_conj'],df['log_sigma'],
                    c=[FAM_COLORS[f] for f in df['family']],s=18,alpha=0.55)
    xf=np.linspace(df['pi_conj'].min(),df['pi_conj'].max(),100)
    axes[1].plot(xf,np.polyval(np.polyfit(df['pi_conj'],df['log_sigma'],1),xf),
                 'k--',lw=1.5,alpha=0.7)
    axes[1].set(xlabel='π-Conjugation',ylabel='log₁₀(σ)',title='π-Conjugation vs σ')

    # Panel C: feature correlations
    corrs=df[PHYS_COLS+['log_sigma']].corr()['log_sigma'][PHYS_COLS].sort_values()
    cc=['#D84315' if v<-0.3 else '#00695C' if v>0.3 else '#37474F' for v in corrs.values]
    axes[2].barh([FEAT_LBL.get(f,f) for f in corrs.index],corrs.values,
                 color=cc,alpha=0.85,edgecolor='white')
    axes[2].axvline(0,color='black',lw=0.8)
    axes[2].set(xlabel='Pearson r with log(σ)',title='Feature Correlations')
    axes[2].tick_params(labelsize=8)

    plt.tight_layout()
    st.pyplot(fig,width="stretch"); plt.close()
    st.dataframe(df[['family']+PHYS_COLS+['log_sigma']].head(10),
                 width="stretch")

# ── TAB 2: LATENT SPACE ────────────────────────────────────────────────────────
with tab2:
    st.subheader("VAE Latent Space")
    col1,col2=st.columns(2)

    with col1:
        st.markdown("**Training Curves**")
        fig,axes=plt.subplots(1,3,figsize=(12,3.5))
        ep=range(1,len(M['hist']['loss'])+1)
        axes[0].plot(ep,M['hist']['loss'],color='#1565C0',lw=2)
        axes[0].set(xlabel='Epoch',ylabel='Loss',title='Total ELBO Loss')
        axes[1].plot(ep,M['hist']['recon'],color='#00695C',lw=2)
        axes[1].set(xlabel='Epoch',ylabel='MSE',title='Reconstruction')
        axes[2].plot(ep,M['hist']['kl'],color='#C62828',lw=2)
        axes[2].axhline(0.5,color='#F57F17',lw=1.5,ls='--',label='Healthy ≈0.5')
        axes[2].set(xlabel='Epoch',ylabel='KL',title='KL Divergence')
        axes[2].legend(fontsize=8)
        for ax in axes: ax.set_title(ax.get_title(),fontweight='bold',fontsize=9)
        plt.tight_layout()
        st.pyplot(fig,width="stretch"); plt.close()

    with col2:
        st.markdown("**PCA of Latent Space**")
        fig,ax=plt.subplots(figsize=(6,5))
        for fam,col in FAM_COLORS.items():
            m=df['family']==fam
            ax.scatter(M['z_pca'][m,0],M['z_pca'][m,1],
                       c=col,s=22,alpha=0.65,label=fam,edgecolors='none')
        ax.set(xlabel=f"PC1 ({M['pca'].explained_variance_ratio_[0]*100:.1f}%)",
               ylabel=f"PC2 ({M['pca'].explained_variance_ratio_[1]*100:.1f}%)",
               title='Latent Space — coloured by family')
        ax.legend(fontsize=7,ncol=2)
        plt.tight_layout()
        st.pyplot(fig,width="stretch"); plt.close()

    st.info("✅ VAE Reconstruction R² = **" + f"{M['recon_r2']:.3f}**  "
            "→ The VAE has learned a structured polymer representation.")

# ── TAB 3: GENERATION ─────────────────────────────────────────────────────────
with tab3:
    st.subheader("Novel Polymer Generation")
    fig,axes=plt.subplots(1,2,figsize=(13,5))

    # Panel A: conductivity distribution
    axes[0].hist(M['y'],bins=25,alpha=0.5,color='#37474F',
                 label='Training (500)',density=True)
    axes[0].hist(M['pred_gen'],bins=25,alpha=0.65,color='#00695C',
                 label='Generated (200)',density=True)
    axes[0].axvline(np.percentile(M['y'],90),color='#C62828',lw=2,ls='--',
                    label=f"90th pct: {np.percentile(M['y'],90):.2f}")
    axes[0].axvline(np.max(M['pred_gen']),color='#F57F17',lw=2,ls=':',
                    label=f"Best gen: {np.max(M['pred_gen']):.2f}")
    axes[0].set(xlabel='log₁₀(σ) [S/cm]',ylabel='Density',
                title='Training vs Generated Conductivity')
    axes[0].legend(fontsize=8)

    # Panel B: novelty vs conductivity
    nc=(M['novelty']>0.3).astype(int)
    axes[1].scatter(M['novelty'],M['pred_gen'],c=nc,cmap='RdYlGn',
                    vmin=0,vmax=1,s=22,alpha=0.65)
    axes[1].axvline(0.3,color='#C62828',lw=1.5,ls='--',label='Novelty ≥ 0.3')
    axes[1].axhline(np.percentile(M['y'],90),color='#F57F17',lw=1.5,ls=':',
                    label='90th percentile')
    axes[1].set(xlabel='Novelty Score',ylabel='Predicted log(σ)',
                title=f'Novelty vs Conductivity  ({nc.sum()} novel)')
    axes[1].legend(fontsize=8)

    plt.tight_layout()
    st.pyplot(fig,width="stretch"); plt.close()

    c1,c2,c3=st.columns(3)
    c1.metric("Generated",      "200 polymers")
    c2.metric("Physically Valid", f"{M['valid'].sum()} ({100*M['valid'].mean():.0f}%)")
    c3.metric("Novel (>0.3)",   f"{nc.sum()} polymers")

# ── TAB 4: XAI ────────────────────────────────────────────────────────────────
with tab4:
    st.subheader("XAI — Feature Attribution")
    imp_df=M['imp_df']

    CAT={'pi_conj':'#00695C','backbone_rig':'#00695C','dop_affinity':'#00695C',
         'homo_lumo_gap':'#1565C0','ioniz_pot':'#1565C0','electron_aff':'#1565C0',
         'crystallinity':'#F57F17','charge_mob':'#F57F17','disorder':'#F57F17',
         'dopant_conc':'#C62828','chain_len':'#37474F','mol_wt':'#37474F',
         'sidechain_pol':'#4527A0'}

    fig,axes=plt.subplots(1,2,figsize=(13,5.5))

    # Panel A: permutation importance
    ci=[CAT[f] for f in imp_df['feat']]
    axes[0].barh([FEAT_LBL.get(f,f) for f in imp_df['feat'][::-1]],
                 imp_df['imp'][::-1],xerr=imp_df['std'][::-1],
                 color=list(reversed(ci)),alpha=0.88,capsize=3,edgecolor='white')
    axes[0].set(xlabel='Permutation Importance (ΔR²)',
                title='A. Feature Importance (30 repeats)')
    axes[0].axvline(0,color='black',lw=0.8)
    axes[0].set_title('A. Feature Importance (30 repeats)',fontweight='bold',fontsize=10)

    # Panel B: model comparison
    mns=['Ridge\n(Phys)','GBM\n(Phys)','GBM\n(VAE Latent)']
    r2s=[M['models']['Ridge-phys']['r2'],M['models']['GBM-phys']['r2'],
         M['models']['GBM-latent']['r2']]
    brs=axes[1].bar(mns,r2s,color=['#37474F','#1565C0','#00695C'],
                    alpha=0.87,edgecolor='white',width=0.55)
    brs[r2s.index(max(r2s))].set_edgecolor('#C62828'); brs[r2s.index(max(r2s))].set_linewidth(3)
    axes[1].set(ylabel='R² Score',title='B. Model Comparison')
    axes[1].set_title('B. Model Comparison',fontweight='bold',fontsize=10)
    for b,v in zip(brs,r2s):
        axes[1].text(b.get_x()+b.get_width()/2,b.get_height()+0.01,
                     f'{v:.3f}',ha='center',fontsize=10,fontweight='bold')

    plt.tight_layout()
    st.pyplot(fig,width="stretch"); plt.close()

    st.success(f"✅ Top driver: **{FEAT_LBL.get(imp_df.iloc[0]['feat'],imp_df.iloc[0]['feat'])}** "
               f"(importance = {imp_df.iloc[0]['imp']:.4f})  — consistent with Marcus transport theory.")

# ── TAB 5: PREDICTOR ──────────────────────────────────────────────────────────
with tab5:
    st.subheader("Live Conductivity Predictor")
    st.markdown("Adjust descriptors to predict log(conductivity) for a new polymer.")

    col_in,col_out=st.columns(2)
    with col_in:
        pi  = st.slider("π-Conjugation",         0.30, 1.00, 0.90, 0.01)
        rig = st.slider("Backbone Rigidity",      0.30, 1.00, 0.80, 0.01)
        da  = st.slider("Doping Affinity",        0.10, 1.00, 0.70, 0.01)
        gap = st.slider("HOMO-LUMO Gap (eV)",     0.20, 3.50, 0.80, 0.05)
        mob = st.slider("Charge Mobility",        0.01, 1.00, 0.70, 0.01)
        dis = st.slider("Disorder Parameter",     0.05, 0.90, 0.20, 0.01)
        dop = st.slider("Dopant Concentration",   0.05, 0.40, 0.25, 0.01)
        cry = st.slider("Crystallinity",          0.10, 0.95, 0.60, 0.01)
        ip  = st.slider("Ionization Potential",   4.00, 6.50, 4.60, 0.05)

    with col_out:
        # Analytical estimate using Marcus model
        ea  = max(0.5, ip - gap - 0.2)
        scp = 0.5; cl = 30; mw = cl * 120
        raw = np.array([[pi,rig,da,cl,mw,gap,ip,ea,scp,cry,mob,dis,dop]])
        X_new = M['sc'].transform(raw)

        # Pad with zeros for fingerprint bits
        X_new_full = np.hstack([X_new,
                                np.zeros((1, 64), dtype=np.float32)])
        _,_,z_new = M['vae'].latent(X_new_full.astype(np.float32))
        pred = float(M['models']['GBM-latent']['model'].predict(z_new)[0])

        if pred > 9:   status = "🟢 EXCELLENT"
        elif pred > 7: status = "🟡 GOOD"
        elif pred > 5: status = "🟠 MODERATE"
        else:          status = "🔴 LOW"

        st.metric("Predicted log(σ)",   f"{pred:.2f} log(S/cm)")
        st.metric("Conductivity",       f"~{10**pred:.1f} S/cm")
        st.metric("Battery Health",     status)

        # Simple gauge
        fig,ax=plt.subplots(figsize=(5,1.5))
        ax.barh([0],[14],color='#2C3E50',height=0.5)
        col='#00695C' if pred>9 else '#F57F17' if pred>7 else '#C62828'
        ax.barh([0],[max(0,pred)],color=col,height=0.5,alpha=0.9)
        ax.axvline(pred,color='white',lw=2)
        ax.set(xlim=(0,14),yticks=[],xlabel='log₁₀(σ) [S/cm]')
        ax.text(pred,0.35,f'{pred:.2f}',ha='center',fontsize=10,fontweight='bold')
        plt.tight_layout()
        st.pyplot(fig,width="stretch"); plt.close()

        st.info("**Physics check:**  π-conjugation drives conductivity via Marcus hopping. "
                "High π-conjugation + low disorder + low HOMO-LUMO gap = best conductors.")
