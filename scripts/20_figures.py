"""Revised figures 1, 3, 9, 10, 11 and 12 (300 dpi). Requires results of 02 and 04; Fig. 1 also needs
data/point_metadata.csv (script 12)."""
import warnings; warnings.filterwarnings("ignore")
from common import *
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pyproj import Transformer
from pykrige.ok import OrdinaryKriging
from scipy.spatial import ConvexHull
from matplotlib.path import Path
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.titlesize':13,'axes.labelsize':11.5,'xtick.labelsize':10,'ytick.labelsize':10,'legend.fontsize':10,'axes.edgecolor':'#555','axes.linewidth':0.8})
d=load_dataset()
o=pd.read_csv(RESULTS/'oof_predictions_hotspots.csv'); assert (o.Point_ID.values==d.Point_ID.values).all(); d['oof']=o.E_total_RF_OOF.values
META=DATA/'point_metadata.csv'
if META.exists(): d['ptype']=pd.read_csv(META).Location_type.values
ANT=load_base_stations()
ANT['label']=ANT.BS_label; ANT['sq']=ANT.Square
CM='OrRd'; vmin,vmax=d.E_total.min(),d.E_total.max()
NAMES=SQUARE_NAMES
def setup(ax,g,pad=25):
    ax.set_aspect('equal')
    x0,x1=g.X.min()-pad,g.X.max()+pad; y0,y1=g.Y.min()-pad,g.Y.max()+pad
    a=ANT[ANT.sq==g.Square.iloc[0]]
    x0=min(x0,a.X.min()-pad);x1=max(x1,a.X.max()+pad);y0=min(y0,a.Y.min()-pad);y1=max(y1,a.Y.max()+pad)
    ax.set_xlim(x0,x1);ax.set_ylim(y0,y1)
    ax.ticklabel_format(useOffset=False,style='plain')
    ax.set_xlabel('Easting, UTM 34N (m)');ax.set_ylabel('Northing, UTM 34N (m)')
    ax.tick_params(axis='x',rotation=30)
    ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(5));ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(6))
    ax.grid(color='#ddd',lw=0.5,zorder=0)
    # scale bar + north arrow
    L=50 if (x1-x0)<250 else 100
    sx=x0+0.05*(x1-x0); sy=y0+0.05*(y1-y0)
    ax.plot([sx,sx+L],[sy,sy],color='k',lw=3,solid_capstyle='butt',zorder=6)
    ax.text(sx+L/2,sy+0.02*(y1-y0),f'{L} m',ha='center',va='bottom',fontsize=10,zorder=6)
    ax.annotate('N',xy=(x1-0.06*(x1-x0),y1-0.04*(y1-y0)),xytext=(x1-0.06*(x1-x0),y1-0.16*(y1-y0)),ha='center',va='center',fontsize=12,fontweight='bold',arrowprops=dict(arrowstyle='-|>',color='k',lw=1.5),zorder=6)
def ants(ax,sq,labels=True):
    a=ANT[ANT.sq==sq]
    ax.scatter(a.X,a.Y,marker='^',s=170,c='#1f4e9c',edgecolors='white',linewidths=1.2,zorder=8)
    if labels:
        OFF={'BS-M2':(-62,-18),'BS-M3':(-62,8),'BS-M1':(7,7),'BS-K1':(7,7),'BS-K2':(7,7)}
        for _,r in a.iterrows(): ax.annotate(r.label,(r.X,r.Y),xytext=OFF[r.label],textcoords='offset points',fontsize=10,fontweight='bold',color='#1f4e9c',zorder=9,bbox=dict(fc='white',ec='none',alpha=0.8,pad=1))
def hullmask(g,XX,YY):
    pts=g[['X','Y']].values; h=ConvexHull(pts); return Path(pts[h.vertices]).contains_points(np.c_[XX.ravel(),YY.ravel()]).reshape(XX.shape)
def krige(g,col,res=2.0):
    x0,x1,y0,y1=g.X.min(),g.X.max(),g.Y.min(),g.Y.max()
    gx=np.arange(x0,x1+res,res);gy=np.arange(y0,y1+res,res)
    rng=np.random.default_rng(0)
    ok=OrdinaryKriging(g.X.values+rng.normal(0,0.05,len(g)),g.Y.values+rng.normal(0,0.05,len(g)),g[col].values,variogram_model='spherical',nlags=10)
    z,ss=ok.execute('grid',gx,gy); XX,YY=np.meshgrid(gx,gy); m=hullmask(g,XX,YY)
    return XX,YY,np.where(m,z,np.nan),np.where(m,np.sqrt(ss),np.nan),ok
mk={'Contour':'o','Urban':'s','Cluster':'D'}
# ---------- Fig 1
if 'ptype' in d:
  fig,axs=plt.subplots(1,2,figsize=(14,7.2),gridspec_kw=dict(width_ratios=[1,1.35]))
  for ax,sq,lab in zip(axs,['MKA','KT'],['(a)','(b)']):
      g=d[d.Square==sq]; setup(ax,g)
      for t,gg in g.groupby('ptype'):
          sc=ax.scatter(gg.X,gg.Y,c=gg.E_total,cmap=CM,vmin=vmin,vmax=vmax,marker=mk[t],s=70,edgecolors='#333',linewidths=0.6,zorder=5)
      ants(ax,sq)
      ax.set_title(f'{lab} {NAMES[sq]} (n = {len(g)})',loc='left')
  cb=fig.colorbar(sc,ax=axs,shrink=0.85,pad=0.02); cb.set_label(r'$E_{total}$ (V/m)')
  h=[Line2D([],[],marker=mk[t],ls='',mfc='#bbb',mec='#333',ms=9,label=f'{t} point') for t in ['Cluster','Contour','Urban']]+[Line2D([],[],marker='^',ls='',mfc='#1f4e9c',mec='white',ms=13,label='Base station (BS)')]
  fig.legend(handles=h,loc='lower center',ncol=4,frameon=False,bbox_to_anchor=(0.45,-0.01))
  fig.savefig(FIGURES/'Figure_01_Study_Areas_Points_Base_Stations.png',dpi=300,bbox_inches='tight');plt.close()
else: print("Fig. 1 skipped: data/point_metadata.csv missing (run 12_point_metadata.py)")
# ---------- Fig 3 workflow (vertical, large fonts)
fig,ax=plt.subplots(figsize=(8.5,10));ax.axis('off');ax.set_xlim(0,100);ax.set_ylim(0,100)
boxes=[('Field measurements','171 points (85 MKA, 86 KT), Narda SRM-3006\n700 MHz–6 GHz sweep, 531 bins per point'),
       ('Band features (7 predictors)','RSS within the 3GPP allocations of\nLTE700, LTE800, GSM900, GSM1800, LTE2100, LTE2600, NR3500'),
       ('Spatial features (4 predictors)','Latitude, longitude (WGS84),\ndistance to nearest base station, antenna height'),
       ('Target variable','$E_{total}$ = RSS of the full 700 MHz–6 GHz sweep (Eq. 1)'),
       ('Final dataset','171 samples × 11 predictors → Random Forest and MLR\ntraining, validation and spatial mapping')]
h=15.5;gap=4.4
for i,(t,bd) in enumerate(boxes):
    y=100-(i+1)*h-i*gap
    ax.add_patch(matplotlib.patches.FancyBboxPatch((3,y),94,h,boxstyle='round,pad=0.3,rounding_size=1.5',fc='#f2f2f2',ec='#333',lw=1.3))
    ax.text(50,y+h-4.2,t,ha='center',va='center',fontsize=15,fontweight='bold')
    ax.text(50,y+5.2,bd,ha='center',va='center',fontsize=12.5,linespacing=1.4)
    if i<4: ax.annotate('',xy=(50,y-gap+0.3),xytext=(50,y-0.4),arrowprops=dict(arrowstyle='-|>',lw=2,color='#333'))
fig.savefig(FIGURES/'Figure_03_Dataset_Construction_Workflow.png',dpi=300,bbox_inches='tight');plt.close()
# ---------- Fig 9 kriged measured vs predicted
fig,axs=plt.subplots(2,2,figsize=(13,12.5),gridspec_kw=dict(width_ratios=[1,1.35],hspace=0.32))
for j,sq in enumerate(['MKA','KT']):
    g=d[d.Square==sq]
    for i,(col,ttl) in enumerate([('E_total','Measured'),('oof','RF-predicted (out-of-fold)')]):
        ax=axs[i,j]; setup(ax,g,pad=15)
        XX,YY,Z,_,_=krige(g,col)
        cf=ax.contourf(XX,YY,Z,levels=np.linspace(vmin,vmax,25),cmap=CM,vmin=vmin,vmax=vmax,extend='both',zorder=2)
        ax.scatter(g.X,g.Y,s=12,c='k',zorder=5)
        ants(ax,sq,labels=True)
        ax.set_title(f"({'abcd'[i*2+j]}) {ttl}\n{NAMES[sq]}",loc='left',fontsize=12)
cb=fig.colorbar(cf,ax=axs,shrink=0.6,pad=0.02);cb.set_label(r'$E_{total}$ (V/m)');cb.set_ticks(np.round(np.linspace(3,7,9),1))
fig.savefig(FIGURES/'Figure_09_Measured_vs_RF_OOF_Kriged_Maps.png',dpi=300,bbox_inches='tight');plt.close()
# ---------- Fig 10 kriged measured with P85
fig,axs=plt.subplots(1,2,figsize=(14,6.8),gridspec_kw=dict(width_ratios=[1.35,1]))
sdr={}
for j,sq in enumerate(['KT','MKA']):
    g=d[d.Square==sq]; p85=np.percentile(g.E_total,85)
    XX,YY,Z,SD,ok=krige(g,'E_total'); sdr[sq]=(np.nanmin(SD),np.nanmax(SD))
    ax=axs[j];setup(ax,g,pad=15)
    cf=ax.contourf(XX,YY,Z,levels=np.linspace(vmin,vmax,25),cmap=CM,vmin=vmin,vmax=vmax,extend='both',zorder=2)
    ax.contour(XX,YY,Z,levels=[p85],colors='k',linewidths=2,zorder=4)
    hs=g.E_total>=p85
    ax.scatter(g.X[~hs],g.Y[~hs],s=22,c='white',edgecolors='k',linewidths=0.6,zorder=5)
    ax.scatter(g.X[hs],g.Y[hs],s=80,marker='^',c='k',edgecolors='white',linewidths=0.6,zorder=6)
    ants(ax,sq)
    ax.set_title(f"({'ab'[j]}) {NAMES[sq]}: P85 = {p85:.2f} V/m",loc='left',fontsize=12)
print('kriging SD ranges',sdr)
cb=fig.colorbar(cf,ax=axs,shrink=0.8,pad=0.02);cb.set_label(r'Measured $E_{total}$ (V/m)');cb.set_ticks([3,3.5,4,4.5,5,5.5,6,6.5,7])
h=[Line2D([],[],color='k',lw=2,label='P85 contour'),Line2D([],[],marker='^',ls='',mfc='k',mec='white',ms=10,label='Hotspot point (≥ P85)'),Line2D([],[],marker='o',ls='',mfc='white',mec='k',ms=7,label='Other measurement point'),Line2D([],[],marker='^',ls='',mfc='#1f4e9c',mec='white',ms=12,label='Base station')]
fig.legend(handles=h,loc='lower center',ncol=4,frameon=False,bbox_to_anchor=(0.45,-0.06))
fig.savefig(FIGURES/'Figure_10_Kriged_Measured_P85_Hotspots.png',dpi=300,bbox_inches='tight');plt.close()
# ---------- Fig 11 hotspot agreement
cls_style={'TP':('#c0392b','o','Hotspot in both (TP)'),'FN':('#1f77b4','s','Measured only, missed (FN)'),'FP':('#e69f00','D','Model only, false alarm (FP)'),'TN':('#c8c8c8','o','Neither (TN)')}
fig,axs=plt.subplots(1,2,figsize=(14,6.6),gridspec_kw=dict(width_ratios=[1.35,1]))
for ax,sq,lab in zip(axs,['KT','MKA'],'ab'):
    g=d[d.Square==sq].copy(); mt=g.E_total>=np.percentile(g.E_total,85); pt=g.oof>=np.percentile(g.oof,85)
    g['c']=np.select([mt&pt,mt&~pt,~mt&pt],['TP','FN','FP'],'TN'); setup(ax,g,pad=15)
    for c in ['TN','FP','FN','TP']:
        gg=g[g.c==c]; col,m,_=cls_style[c]
        ax.scatter(gg.X,gg.Y,c=col,marker=m,s=30 if c=='TN' else 85,edgecolors='#222' if c!='TN' else 'none',linewidths=0.6,zorder=5)
    ants(ax,sq); ax.set_title(f'({lab}) {NAMES[sq]}',loc='left')
    print(sq,g.c.value_counts().to_dict())
h=[Line2D([],[],marker=cls_style[c][1],ls='',mfc=cls_style[c][0],mec='#222',ms=9,label=cls_style[c][2]) for c in ['TP','FN','FP','TN']]+[Line2D([],[],marker='^',ls='',mfc='#1f4e9c',mec='white',ms=12,label='Base station')]
fig.legend(handles=h,loc='lower center',ncol=3,frameon=False,bbox_to_anchor=(0.5,-0.13))
fig.savefig(FIGURES/'Figure_11_Hotspot_Agreement.png',dpi=300,bbox_inches='tight');plt.close()
# ---------- Fig 12 learning curve
lc=pd.read_csv(RESULTS/'learning_curves_raw.csv')
fig,ax=plt.subplots(figsize=(8,5))
st={'MLR (11 features)':('#1f4e9c','s'),'RF (11 features)':('#c0392b','o'),'RF (spatial only)':('#7f7f7f','^')}
for m,(col,mkr) in st.items():
    s=lc[lc.model==m].groupby('n').r2; md=s.median(); q1=s.quantile(.25); q3=s.quantile(.75)
    ax.fill_between(md.index,q1,q3,color=col,alpha=0.15,lw=0); ax.plot(md.index,md,color=col,marker=mkr,lw=2,ms=8,label=m)
    ax.annotate(m,(md.index[-1],md.iloc[-1]),xytext=(6,0),textcoords='offset points',va='center',fontsize=10,color='#333')
ax.set_xlabel('Number of training points');ax.set_ylabel('Test-set R² (median, IQR band)')
ax.set_xlim(10,175);ax.set_ylim(-0.15,1.0);ax.grid(color='#e5e5e5',lw=0.6);ax.axhline(0,color='#999',lw=0.8)
ax.legend(loc='lower right',frameon=False)
for s in ['top','right']: ax.spines[s].set_visible(False)
fig.savefig(FIGURES/'Figure_12_Learning_Curves.png',dpi=300,bbox_inches='tight');plt.close()
print('kriging done')
