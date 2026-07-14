"""
AI-based Regional Aftershock Prediction
Physics-Guided Relational Representation (PGRR)
Sumatra Subduction Zone (2000-2025)
Models: Random Forest, SVM, Logistic Regression
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import warnings
warnings.filterwarnings("ignore")

from core.physics import (
    haversine_km, label_aftershocks_magnitude_scaled,
    build_history_features_phys, detect_columns,
    FEATURE_COLS, FEATURE_PRETTY, LABEL_COL,
    SUMATRA_BBOX, R_of_M_km, T_of_M_days,
)
from core.models import (
    MODEL_REGISTRY, build_rf_pipeline,
    evaluate_model, best_threshold_f1, bootstrap_ci_auc,
    get_feature_importance, ABLATION_SETS, ABLATION_LABELS,
)

st.set_page_config(page_title="Prediksi Gempa Susulan — PGRR Sumatra", page_icon="🌏", layout="wide", initial_sidebar_state="expanded")

# ── CSS ──
st.markdown("""<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');
.main .block-container{padding-top:1rem;max-width:1400px}
.hero-header{background:linear-gradient(135deg,#0a0e17 0%,#1a2332 50%,#0d1b2a 100%);border:1px solid #2a3444;border-radius:16px;padding:2rem 2.5rem;margin-bottom:1.5rem;position:relative;overflow:hidden}
.hero-header::before{content:'';position:absolute;top:0;left:0;right:0;height:3px;background:linear-gradient(90deg,#06d6a0,#118ab2,#ef476f,#f77f00)}
.hero-title{font-family:'Plus Jakarta Sans',sans-serif;font-weight:800;font-size:2rem;background:linear-gradient(135deg,#06d6a0,#118ab2);-webkit-background-clip:text;-webkit-text-fill-color:transparent;margin:0 0 .5rem 0}
.hero-subtitle{font-family:'Plus Jakarta Sans',sans-serif;color:#8892a4;font-size:1rem}
.metric-card{background:linear-gradient(145deg,#1a2332,#111827);border:1px solid #2a3444;border-radius:12px;padding:1.2rem 1.5rem;text-align:center;transition:border-color .3s}
.metric-card:hover{border-color:#06d6a0}
.metric-value{font-family:'JetBrains Mono',monospace;font-size:1.8rem;font-weight:700;color:#06d6a0}
.metric-label{font-family:'Plus Jakarta Sans',sans-serif;font-size:.85rem;color:#8892a4;margin-top:.3rem}
.tag-physics{display:inline-block;background:rgba(6,214,160,.15);color:#06d6a0;border:1px solid rgba(6,214,160,.3);border-radius:6px;padding:.2rem .6rem;font-size:.8rem;font-family:'JetBrains Mono',monospace;margin-right:.4rem;margin-bottom:.3rem}
.tag-ml{display:inline-block;background:rgba(17,138,178,.15);color:#118ab2;border:1px solid rgba(17,138,178,.3);border-radius:6px;padding:.2rem .6rem;font-size:.8rem;font-family:'JetBrains Mono',monospace;margin-right:.4rem;margin-bottom:.3rem}
.section-header{font-family:'Plus Jakarta Sans',sans-serif;font-weight:700;font-size:1.4rem;color:#e8ecf1;border-left:4px solid #06d6a0;padding-left:1rem;margin:1.5rem 0 1rem 0}
.info-box{background:#111827;border:1px solid #2a3444;border-radius:10px;padding:1rem 1.2rem;margin-bottom:.8rem}
.info-box-title{font-family:'Plus Jakarta Sans',sans-serif;font-weight:600;color:#e8ecf1;font-size:1rem}
.info-box-desc{font-family:'Plus Jakarta Sans',sans-serif;color:#8892a4;font-size:.88rem;line-height:1.6}
.param-help{font-size:.82rem;color:#8892a4;font-style:italic;margin-top:-.5rem;margin-bottom:.8rem}
</style>""", unsafe_allow_html=True)

# ── Session State ──
for k, v in {"df_raw":None,"df_labeled":None,"df_features":None,"train_model":None,"val_model":None,"test_model":None,"trained_models":{},"eval_results":{},"mc_value":4.45,"t_cap":365}.items():
    if k not in st.session_state: st.session_state[k] = v

COLORS = {"cyan":"#06d6a0","blue":"#118ab2","red":"#ef476f","orange":"#f77f00","purple":"#8338ec"}
FEATURE_FRIENDLY = {"mag":"Kekuatan Gempa (Magnitudo)","log_dt_big_near":"Waktu sejak gempa besar terakhir","log_dr_big_near":"Jarak dari gempa besar terakhir","log_n_prev_30d":"Jumlah gempa 30 hari terakhir","log_n_prev_30d_r50":"Jumlah gempa dekat (50 km) 30 hari","log_n_big_prev_30d":"Jumlah gempa besar 30 hari terakhir","max_mag_prev_7d":"Gempa terbesar 7 hari terakhir"}
def metric_html(value, label, color="#06d6a0"):
    return f'<div class="metric-card"><div class="metric-value" style="color:{color}">{value}</div><div class="metric-label">{label}</div></div>'

# ── Header ──
st.markdown("""<div class="hero-header"><div class="hero-title">🌏 AI-based Regional Aftershock Prediction</div><div class="hero-subtitle">Prediksi Gempa Susulan Wilayah Sumatra menggunakan Physics-Guided Relational Representation (PGRR)</div><div style="margin-top:.8rem"><span class="tag-physics">Pola Peluruhan Waktu</span><span class="tag-physics">Kedekatan Jarak</span><span class="tag-physics">Kepadatan Aktivitas</span><span class="tag-ml">Random Forest</span><span class="tag-ml">SVM</span><span class="tag-ml">Logistic Regression</span></div></div>""", unsafe_allow_html=True)

# ── Sidebar ──
with st.sidebar:
    st.markdown("### 📌 Menu Utama")
    page = st.radio("Pilih halaman:",[
        "📥 1. Upload & Jelajahi Data","🏷️ 2. Tandai Gempa Susulan","🔬 3. Hitung Fitur Prediksi",
        "🤖 4. Latih Model AI","📊 5. Evaluasi & Perbandingan","🔮 6. Prediksi Gempa Baru","📐 7. Uji Kontribusi Fitur"],label_visibility="collapsed")
    st.markdown("---")
    st.markdown("### 📋 Status Proses")
    for label, done in {"Data Dimuat":st.session_state.df_raw is not None,"Gempa Susulan Ditandai":st.session_state.df_labeled is not None,"Fitur Dihitung":st.session_state.df_features is not None,"Model Terlatih":len(st.session_state.trained_models)>0}.items():
        st.markdown(f"{'✅' if done else '⬜'} {label}")
    st.markdown("---")
    st.markdown("<div style='text-align:center;color:#8892a4;font-size:.8rem'>Prediksi Gempa Susulan<br>PGRR · Sumatra · 2000–2025</div>",unsafe_allow_html=True)

# ═══════ PAGE 1: UPLOAD DATA ═══════
if page.startswith("📥"):
    st.markdown('<div class="section-header">Upload & Jelajahi Data Gempa</div>',unsafe_allow_html=True)
    st.markdown('<div class="info-box"><span class="info-box-title">💡 Apa yang dilakukan di halaman ini?</span><br><span class="info-box-desc">Anda mengupload data katalog gempa bumi (file CSV). Sistem akan otomatis mengenali kolom-kolom penting seperti waktu, lokasi, dan kekuatan gempa, lalu menampilkan ringkasan dan visualisasi.</span></div>',unsafe_allow_html=True)
    import os
    SAMPLE_DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "data_gempa_sumatera.csv")
    has_sample = os.path.exists(SAMPLE_DATA_PATH)

    # ── Download sample data section ──
    if has_sample:
        st.markdown("""<div class="info-box" style="border-color:#118ab2">
        <span class="info-box-title">📥 Data Contoh Tersedia!</span><br>
        <span class="info-box-desc">
            Aplikasi ini sudah menyertakan <b>dataset gempa bumi Sumatra (2000–2025)</b> dari USGS sebanyak ~17.900 event.
            Anda bisa langsung menggunakannya atau mengunduhnya terlebih dahulu untuk dilihat.
        </span></div>""", unsafe_allow_html=True)
        c_dl1, c_dl2 = st.columns(2)
        with c_dl1:
            with open(SAMPLE_DATA_PATH, "rb") as f:
                st.download_button("📥 Unduh Dataset Sumatra (CSV, ~3 MB)", f, file_name="data_gempa_sumatera.csv", mime="text/csv", use_container_width=True)
        with c_dl2:
            use_bundled = st.button("🚀 Langsung Gunakan Dataset Sumatra", type="primary", use_container_width=True)
        if use_bundled:
            df = pd.read_csv(SAMPLE_DATA_PATH); st.session_state.df_raw = df
            st.success(f"✅ Dataset Sumatra dimuat: **{df.shape[0]:,}** gempa × **{df.shape[1]}** kolom")
        st.markdown("---")
        st.markdown("**Atau upload data gempa Anda sendiri:**")

    uploaded = st.file_uploader("Upload file CSV katalog gempa",type=["csv"],help="File harus memiliki kolom: time, latitude, longitude, mag")
    if uploaded is not None:
        df = pd.read_csv(uploaded); st.session_state.df_raw = df
        st.success(f"✅ Data berhasil dimuat: **{df.shape[0]:,}** gempa × **{df.shape[1]}** kolom")
    if st.session_state.df_raw is not None:
        df = st.session_state.df_raw; detected = detect_columns(df)
        st.markdown("#### 🔍 Kolom yang Terdeteksi"); st.json({k:v for k,v in detected.items() if v is not None})
        c1,c2,c3,c4 = st.columns(4)
        with c1: st.markdown(metric_html(f"{len(df):,}","Total Gempa"),unsafe_allow_html=True)
        mag_col = detected.get("mag")
        if mag_col:
            with c2: st.markdown(metric_html(f"{df[mag_col].min():.1f} – {df[mag_col].max():.1f}","Rentang Magnitudo"),unsafe_allow_html=True)
        time_col = detected.get("time")
        if time_col:
            df[time_col] = pd.to_datetime(df[time_col],errors="coerce",utc=True)
            with c3: st.markdown(metric_html(f"{df[time_col].dt.year.min()}–{df[time_col].dt.year.max()}","Rentang Tahun"),unsafe_allow_html=True)
        with c4: st.markdown(metric_html(f"{df.isna().sum().sum():,}","Data Kosong",COLORS["orange"]),unsafe_allow_html=True)
        st.markdown("#### 📋 Tabel Data"); st.dataframe(df.head(20),use_container_width=True,height=300)
        if mag_col:
            st.markdown("#### 📊 Sebaran Kekuatan Gempa")
            fig=px.histogram(df,x=mag_col,nbins=50,color_discrete_sequence=[COLORS["cyan"]],template="plotly_dark",labels={mag_col:"Magnitudo"})
            fig.update_layout(height=350,paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(17,24,39,0.6)")
            st.plotly_chart(fig,use_container_width=True)
        lat_col,lon_col = detected.get("lat"),detected.get("lon")
        if lat_col and lon_col:
            st.markdown("#### 🗺️ Peta Lokasi Gempa")
            fig=px.scatter_mapbox(df.dropna(subset=[lat_col,lon_col]).sample(min(5000,len(df))),lat=lat_col,lon=lon_col,color=mag_col,color_continuous_scale="Turbo",size_max=8,zoom=3,mapbox_style="carto-darkmatter",height=500)
            fig.update_layout(margin=dict(l=0,r=0,t=0,b=0),paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig,use_container_width=True)

# ═══════ PAGE 2: LABELING ═══════
elif page.startswith("🏷"):
    st.markdown('<div class="section-header">Tandai Gempa Susulan (Aftershock Labeling)</div>',unsafe_allow_html=True)
    if st.session_state.df_raw is None: st.warning("⚠️ Upload data terlebih dahulu di halaman 1.")
    else:
        st.markdown('<div class="info-box"><span class="info-box-title">💡 Apa yang dilakukan di halaman ini?</span><br><span class="info-box-desc">Sistem menentukan gempa mana yang merupakan <b>gempa susulan</b> dan mana yang <b>gempa utama</b>. Jika sebuah gempa terjadi <b>dekat</b> dan <b>segera setelah</b> gempa besar sebelumnya, maka dianggap gempa susulan. Semakin besar gempa utamanya, semakin luas area dan waktu yang diperiksa.</span></div>',unsafe_allow_html=True)
        c1,c2 = st.columns(2)
        with c1: mc_val = st.number_input("Magnitudo minimum yang dianalisis (Mc)",value=4.45,min_value=1.0,max_value=8.0,step=0.05,help="Gempa di bawah nilai ini diabaikan karena pencatatannya tidak lengkap")
        with c2: t_cap = st.selectbox("Batas waktu maksimum susulan (hari)",[180,365,730],index=1,help="Setelah berapa hari gempa tidak lagi dianggap susulan. 365 = 1 tahun")
        st.session_state.mc_value = mc_val; st.session_state.t_cap = t_cap
        with st.expander("📐 Detail teknis: bagaimana jarak dan waktu dihitung?"):
            st.markdown("**Jangkauan jarak:** Semakin besar gempa utama → efeknya terasa semakin jauh"); st.latex(r"R(M) = 10^{0.5M - 1.5} \text{ km}")
            st.markdown("**Jangkauan waktu:** Semakin besar gempa utama → susulan bisa terjadi lebih lama"); st.latex(r"T(M) = \min(10^{0.5M - 1.0}, T_{cap}) \text{ hari}")
            ex = [{"Magnitudo":m,"Jarak (km)":f"{R_of_M_km(m):.1f}","Waktu (hari)":f"{T_of_M_days(m,t_cap):.1f}"} for m in [4.5,5.0,5.5,6.0,7.0,8.0,9.0]]
            st.dataframe(pd.DataFrame(ex),use_container_width=True,hide_index=True)
        if st.button("🚀 Mulai Penandaan",type="primary",use_container_width=True):
            df_raw = st.session_state.df_raw.copy(); detected = detect_columns(df_raw)
            rn = {}
            if detected["time"] and detected["time"]!="time": rn[detected["time"]]="time"
            if detected["lat"] and detected["lat"]!="latitude": rn[detected["lat"]]="latitude"
            if detected["lon"] and detected["lon"]!="longitude": rn[detected["lon"]]="longitude"
            if detected["mag"] and detected["mag"]!="mag": rn[detected["mag"]]="mag"
            df_raw = df_raw.rename(columns=rn); df_raw["time"]=pd.to_datetime(df_raw["time"],errors="coerce",utc=True)
            df_mc = df_raw[df_raw["mag"]>=mc_val].copy()
            with st.spinner("⏳ Menandai gempa susulan..."):
                df_labeled = label_aftershocks_magnitude_scaled(df_mc,MC_FINAL=mc_val,T_cap_days=t_cap)
            st.session_state.df_labeled = df_labeled
            rate=df_labeled["is_aftershock"].mean(); n_as=int(df_labeled["is_aftershock"].sum())
            c1,c2,c3=st.columns(3)
            with c1: st.markdown(metric_html(f"{len(df_labeled):,}","Total Diproses"),unsafe_allow_html=True)
            with c2: st.markdown(metric_html(f"{n_as:,}","Gempa Susulan",COLORS["red"]),unsafe_allow_html=True)
            with c3: st.markdown(metric_html(f"{rate:.1%}","Persentase",COLORS["orange"]),unsafe_allow_html=True)
            st.success(f"✅ Ditemukan **{n_as:,}** gempa susulan ({rate:.1%})")
        if st.session_state.df_labeled is not None:
            dl = st.session_state.df_labeled
            fig=px.pie(values=[int((dl["is_aftershock"]==0).sum()),int(dl["is_aftershock"].sum())],names=["Gempa Utama/Mandiri","Gempa Susulan"],color_discrete_sequence=[COLORS["blue"],COLORS["red"]],hole=0.5)
            fig.update_layout(height=300,paper_bgcolor="rgba(0,0,0,0)",font=dict(color="#e8ecf1"))
            st.plotly_chart(fig,use_container_width=True)

# ═══════ PAGE 3: FEATURES ═══════
elif page.startswith("🔬"):
    st.markdown('<div class="section-header">Hitung Fitur Prediksi</div>',unsafe_allow_html=True)
    if st.session_state.df_labeled is None: st.warning("⚠️ Tandai gempa susulan dahulu di halaman 2.")
    else:
        st.markdown('<div class="info-box"><span class="info-box-title">💡 Apa yang dilakukan?</span><br><span class="info-box-desc">Untuk setiap gempa, dihitung <b>7 karakteristik penting</b>:<br>• <b>Pola Waktu:</b> Seberapa baru gempa besar terjadi?<br>• <b>Pola Jarak:</b> Seberapa dekat dari gempa besar?<br>• <b>Pola Kepadatan:</b> Seberapa ramai aktivitas gempa akhir-akhir ini?</span></div>',unsafe_allow_html=True)
        if st.button("🔧 Hitung Fitur Prediksi",type="primary",use_container_width=True):
            with st.spinner("⏳ Menghitung..."):
                df_feat = build_history_features_phys(st.session_state.df_labeled)
                df_feat["time"]=pd.to_datetime(df_feat["time"],utc=True)
                te=pd.to_datetime("2018-12-31",utc=True); ve=pd.to_datetime("2021-12-31",utc=True)
                df_feat["split"]=np.where(df_feat["time"]<=te,"train",np.where(df_feat["time"]<=ve,"val","test"))
                st.session_state.df_features=df_feat
                for s in ["train","val","test"]:
                    st.session_state[f"{s}_model"]=df_feat[df_feat["split"]==s][FEATURE_COLS+[LABEL_COL]].copy()
            c1,c2,c3=st.columns(3)
            for s,lbl,clr in [("train","Data Latih (2000–2018)","#06d6a0"),("val","Data Validasi (2019–2021)","#118ab2"),("test","Data Uji (2022–2025)","#f77f00")]:
                with [c1,c2,c3][["train","val","test"].index(s)]:
                    st.markdown(metric_html(f"{len(st.session_state[f'{s}_model']):,}",lbl,clr),unsafe_allow_html=True)
            st.success("✅ Fitur dihitung & data dibagi berdasarkan waktu!")
        if st.session_state.df_features is not None:
            st.markdown("#### 📈 Perbandingan Fitur: Susulan vs Bukan")
            fp = st.selectbox("Pilih fitur:",FEATURE_COLS,format_func=lambda x:FEATURE_FRIENDLY.get(x,x))
            fig=px.histogram(st.session_state.df_features,x=fp,color="is_aftershock",barmode="overlay",nbins=50,opacity=0.7,color_discrete_map={0:COLORS["blue"],1:COLORS["red"]},labels={"is_aftershock":"Jenis",fp:FEATURE_FRIENDLY.get(fp,fp)},template="plotly_dark")
            fig.update_layout(height=350,paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(17,24,39,0.6)")
            fig.for_each_trace(lambda t:t.update(name="Susulan" if t.name=="1" else "Bukan Susulan"))
            st.plotly_chart(fig,use_container_width=True)

# ═══════ PAGE 4: TRAINING ═══════
elif page.startswith("🤖"):
    st.markdown('<div class="section-header">Latih Model AI</div>',unsafe_allow_html=True)
    if st.session_state.train_model is None: st.warning("⚠️ Hitung fitur dahulu di halaman 3.")
    else:
        tm,vm,tsm = st.session_state.train_model,st.session_state.val_model,st.session_state.test_model
        st.markdown(f'<div class="info-box"><span class="info-box-title">💡 Apa yang dilakukan?</span><br><span class="info-box-desc">Melatih 3 model AI untuk membedakan gempa susulan dari gempa utama. Latih: <b>{len(tm):,}</b> · Validasi: <b>{len(vm):,}</b> · Uji: <b>{len(tsm):,}</b></span></div>',unsafe_allow_html=True)
        sel = st.multiselect("Pilih model:",list(MODEL_REGISTRY.keys()),default=list(MODEL_REGISTRY.keys()))
        if st.button("🚀 Latih Semua Model",type="primary",use_container_width=True):
            Xtr,ytr = tm[FEATURE_COLS].values,tm[LABEL_COL].values
            Xv,yv = vm[FEATURE_COLS].values,vm[LABEL_COL].values
            Xts,yts = tsm[FEATURE_COLS].values,tsm[LABEL_COL].values
            prog=st.progress(0); trained={}; results={}
            for i,name in enumerate(sel):
                prog.progress(int(i/len(sel)*100),text=f"Melatih {name}...")
                pipe=MODEL_REGISTRY[name](); pipe.fit(Xtr,ytr)
                vp=pipe.predict_proba(Xv)[:,1]; thr,_,_,_=best_threshold_f1(yv,vp)
                tp=pipe.predict_proba(Xts)[:,1]; ev=evaluate_model(yts,tp,threshold=thr); ev["val_threshold"]=thr
                ev.update(bootstrap_ci_auc(yts,tp,n_boot=500))
                trained[name]=pipe; results[name]=ev
            prog.progress(100,text="✅ Selesai!")
            st.session_state.trained_models=trained; st.session_state.eval_results=results
            rows=[{"Model":n,"ROC-AUC":f"{r['roc_auc']:.4f}","PR-AUC":f"{r['pr_auc']:.4f}","F1":f"{r['f1_aftershock']:.4f}","Ketepatan":f"{r['precision_aftershock']:.4f}","Kelengkapan":f"{r['recall_aftershock']:.4f}","Ambang":f"{r['val_threshold']:.3f}"} for n,r in results.items()]
            st.dataframe(pd.DataFrame(rows),use_container_width=True,hide_index=True)
            st.success("✅ Model berhasil dilatih!")

# ═══════ PAGE 5: EVALUATION ═══════
elif page.startswith("📊"):
    st.markdown('<div class="section-header">Evaluasi & Perbandingan Model</div>',unsafe_allow_html=True)
    if not st.session_state.eval_results: st.warning("⚠️ Latih model dahulu.")
    else:
        res=st.session_state.eval_results; cl=[COLORS["cyan"],COLORS["blue"],COLORS["red"],COLORS["orange"],COLORS["purple"]]
        st.markdown("#### 📈 Kurva ROC"); st.caption("Semakin melengkung ke kiri atas = semakin bagus")
        fig=go.Figure()
        for i,(n,r) in enumerate(res.items()):
            fpr,tpr=r["roc_curve"]; fig.add_trace(go.Scatter(x=fpr,y=tpr,mode="lines",name=f"{n} (AUC={r['roc_auc']:.3f})",line=dict(width=2.5,color=cl[i%len(cl)])))
        fig.add_trace(go.Scatter(x=[0,1],y=[0,1],mode="lines",line=dict(dash="dash",color="#555"),showlegend=False))
        fig.update_layout(height=450,xaxis_title="Tingkat Kesalahan",yaxis_title="Tingkat Deteksi",paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(17,24,39,0.6)",font=dict(color="#e8ecf1"),legend=dict(x=0.4,y=0.05,bgcolor="rgba(0,0,0,0.4)"))
        st.plotly_chart(fig,use_container_width=True)

        st.markdown("#### 📈 Kurva Precision-Recall"); st.caption("Semakin tinggi = deteksi semakin tepat dan lengkap")
        fig2=go.Figure()
        for i,(n,r) in enumerate(res.items()):
            rc,pr=r["pr_curve"]; fig2.add_trace(go.Scatter(x=rc,y=pr,mode="lines",name=f"{n} (AP={r['pr_auc']:.3f})",line=dict(width=2.5,color=cl[i%len(cl)])))
        fig2.update_layout(height=450,xaxis_title="Kelengkapan (Recall)",yaxis_title="Ketepatan (Precision)",paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(17,24,39,0.6)",font=dict(color="#e8ecf1"),legend=dict(x=0.01,y=0.05,bgcolor="rgba(0,0,0,0.4)"))
        st.plotly_chart(fig2,use_container_width=True)

        st.markdown("#### 🧮 Tabel Kebingungan"); st.caption("Diagonal = prediksi benar")
        cols=st.columns(len(res))
        for i,(n,r) in enumerate(res.items()):
            with cols[i]:
                st.markdown(f"**{n}**"); cm=r["confusion_matrix"]; cmn=cm/cm.sum(axis=1,keepdims=True)
                fig=px.imshow(cmn,text_auto=".2f",x=["Bukan","Susulan"],y=["Bukan","Susulan"],color_continuous_scale="Blues",zmin=0,zmax=1)
                fig.update_layout(height=300,paper_bgcolor="rgba(0,0,0,0)",font=dict(color="#e8ecf1",size=11),margin=dict(l=40,r=20,t=20,b=40))
                st.plotly_chart(fig,use_container_width=True)

        if "Random Forest" in st.session_state.trained_models:
            st.markdown("#### 🌲 Faktor Terpenting (Random Forest)")
            imp=get_feature_importance(st.session_state.trained_models["Random Forest"],FEATURE_COLS)
            if imp is not None:
                imp["Pretty"]=imp["Feature"].map(FEATURE_FRIENDLY)
                fig=px.bar(imp,x="Importance",y="Pretty",orientation="h",color="Importance",color_continuous_scale=["#1a2332","#06d6a0"],template="plotly_dark")
                fig.update_layout(height=350,showlegend=False,yaxis=dict(categoryorder="total ascending"),paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(17,24,39,0.6)",font=dict(color="#e8ecf1"),xaxis_title="Tingkat Kepentingan",yaxis_title="")
                st.plotly_chart(fig,use_container_width=True)

# ═══════ PAGE 6: PREDICTION ═══════
elif page.startswith("🔮"):
    st.markdown('<div class="section-header">Prediksi Gempa Baru</div>',unsafe_allow_html=True)
    if not st.session_state.trained_models: st.warning("⚠️ Latih model dahulu.")
    else:
        tab1,tab2 = st.tabs(["🎯 Prediksi Satu Gempa","📁 Prediksi Batch"])
        with tab1:
            st.markdown('<div class="info-box"><span class="info-box-title">💡 Cara menggunakan</span><br><span class="info-box-desc">Masukkan informasi tentang gempa yang ingin dianalisis. Gunakan slider dengan bahasa yang mudah dipahami — sistem otomatis mengkonversi ke format yang dibutuhkan model.</span></div>',unsafe_allow_html=True)
            st.markdown("#### 📝 Informasi Gempa")
            c1,c2 = st.columns(2)
            with c1:
                st.markdown("**🔴 Kekuatan gempa ini**")
                inp_mag = st.slider("Magnitudo",2.0,9.5,5.0,0.1,help="Kekuatan gempa yang sedang dievaluasi")
                st.markdown("**⏱️ Kapan gempa besar terakhir di dekat sini?**")
                inp_dt = st.slider("Hari sejak gempa besar terakhir",0,1000,20,help="0=hari ini, 30=sebulan lalu, 365=setahun lalu")
                st.markdown(f'<div class="param-help">💡 {inp_dt} hari ≈ {inp_dt/30:.1f} bulan lalu</div>',unsafe_allow_html=True)
                st.markdown("**📏 Seberapa jauh dari gempa besar terakhir?**")
                inp_dr = st.slider("Jarak (km)",0,1000,33,help="10=sangat dekat, 100=cukup jauh, 500=sangat jauh")
                st.markdown("**💪 Gempa terbesar 7 hari terakhir?**")
                inp_mm7 = st.slider("Magnitudo terbesar seminggu ini",2.0,9.5,4.5,0.1)
            with c2:
                st.markdown("**📊 Seberapa ramai aktivitas gempa?**")
                inp_n30 = st.slider("Jumlah gempa 30 hari terakhir",0,3000,20,help="Total gempa sebulan terakhir di wilayah")
                st.markdown("**📍 Gempa dekat lokasi ini?**")
                inp_n30r = st.slider("Gempa dalam radius 50 km (30 hari)",0,500,5,help="Gempa sangat dekat selama sebulan")
                st.markdown("**⚡ Gempa BESAR baru-baru ini?**")
                inp_nb = st.slider("Gempa besar (M≥5) dalam 30 hari",0,50,1)
            eps=1e-6
            feats = np.array([[inp_mag,np.log1p(max(inp_dt,eps)),np.log1p(max(inp_dr,eps)),np.log1p(inp_n30),np.log1p(inp_n30r),np.log1p(inp_nb),inp_mm7]])
            st.markdown("---")
            mc = st.selectbox("Pilih model AI:",list(st.session_state.trained_models.keys()))
            if st.button("🔮 Analisis Gempa Ini",type="primary",use_container_width=True):
                pipe=st.session_state.trained_models[mc]; prob=pipe.predict_proba(feats)[0,1]
                r=st.session_state.eval_results.get(mc,{}); thr=r.get("val_threshold",0.5)
                isas = prob>=thr; label="🔴 GEMPA SUSULAN" if isas else "🟢 GEMPA MANDIRI"; color=COLORS["red"] if isas else COLORS["cyan"]
                st.markdown("---")
                c1,c2,c3=st.columns(3)
                with c1: st.markdown(metric_html(f"{prob:.1%}","Probabilitas Susulan",color),unsafe_allow_html=True)
                with c2: st.markdown(metric_html(label,"Hasil Prediksi",color),unsafe_allow_html=True)
                with c3: st.markdown(metric_html(f"{thr:.3f}","Ambang Keputusan","#8892a4"),unsafe_allow_html=True)
                if isas:
                    st.markdown(f'<div class="info-box" style="border-color:{COLORS["red"]}"><span class="info-box-title">🔴 Interpretasi: Kemungkinan Gempa Susulan</span><br><span class="info-box-desc">Probabilitas <b>{prob:.1%}</b> (di atas ambang {thr:.1%}). Gempa ini kemungkinan dipicu oleh gempa besar sebelumnya. Perlu kewaspadaan terhadap gempa susulan lanjutan.</span></div>',unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="info-box" style="border-color:{COLORS["cyan"]}"><span class="info-box-title">🟢 Interpretasi: Kemungkinan Gempa Mandiri</span><br><span class="info-box-desc">Probabilitas susulan hanya <b>{prob:.1%}</b> (di bawah ambang {thr:.1%}). Gempa ini kemungkinan bukan dipicu gempa besar sebelumnya, melainkan gempa mandiri atau awal sekuens baru.</span></div>',unsafe_allow_html=True)
                fig=go.Figure(go.Indicator(mode="gauge+number",value=prob*100,number=dict(suffix="%",font=dict(size=40)),gauge=dict(axis=dict(range=[0,100]),bar=dict(color=color),bgcolor="#1a2332",borderwidth=0,steps=[dict(range=[0,thr*100],color="rgba(6,214,160,0.15)"),dict(range=[thr*100,100],color="rgba(239,71,111,0.15)")],threshold=dict(line=dict(color="#f77f00",width=3),thickness=0.8,value=thr*100))))
                fig.update_layout(height=280,paper_bgcolor="rgba(0,0,0,0)",font=dict(color="#e8ecf1"),margin=dict(l=30,r=30,t=30,b=10))
                st.plotly_chart(fig,use_container_width=True)
        with tab2:
            st.markdown('<div class="info-box"><span class="info-box-title">📁 Prediksi Banyak Gempa</span><br><span class="info-box-desc">Upload CSV berisi data fitur untuk banyak gempa sekaligus.</span></div>',unsafe_allow_html=True)
            bf=st.file_uploader("Upload CSV",type=["csv"],key="batch")
            if bf:
                bdf=pd.read_csv(bf); st.dataframe(bdf.head(),use_container_width=True)
                mb=st.selectbox("Model:",list(st.session_state.trained_models.keys()),key="bm")
                if st.button("🚀 Prediksi Semua"):
                    pipe=st.session_state.trained_models[mb]; thr=st.session_state.eval_results.get(mb,{}).get("val_threshold",0.5)
                    miss=[c for c in FEATURE_COLS if c not in bdf.columns]
                    if miss: st.error(f"Kolom tidak ditemukan: {miss}")
                    else:
                        probs=pipe.predict_proba(bdf[FEATURE_COLS].values)[:,1]
                        bdf["probabilitas"]=probs; bdf["label"]=(probs>=thr).map({True:"Susulan",False:"Mandiri"})
                        st.dataframe(bdf,use_container_width=True)
                        st.download_button("📥 Download Hasil",bdf.to_csv(index=False),"hasil_prediksi.csv","text/csv")

# ═══════ PAGE 7: ABLATION ═══════
elif page.startswith("📐"):
    st.markdown('<div class="section-header">Uji Kontribusi Fitur</div>',unsafe_allow_html=True)
    if st.session_state.train_model is None: st.warning("⚠️ Hitung fitur dahulu.")
    else:
        st.markdown('<div class="info-box"><span class="info-box-title">💡 Apa yang dilakukan?</span><br><span class="info-box-desc">Menguji <b>seberapa penting setiap fitur</b> dengan menghapusnya satu per satu. Jika performa turun drastis, berarti fitur itu krusial — seperti mencabut komponen mobil untuk tahu mana yang paling penting.</span></div>',unsafe_allow_html=True)
        AF={"FULL":"Semua Fitur (Lengkap)","NO_TEMPORAL":"Tanpa info waktu","NO_SPATIAL":"Tanpa info jarak","NO_DENSITY":"Tanpa info kepadatan","NO_SHORT_MAG":"Tanpa info magnitudo 7 hari"}
        if st.button("🔬 Jalankan Uji",type="primary",use_container_width=True):
            tm,vm,tsm=st.session_state.train_model,st.session_state.val_model,st.session_state.test_model
            from sklearn.metrics import roc_auc_score,average_precision_score
            ar=[]; prog=st.progress(0)
            for i,(name,fl) in enumerate(ABLATION_SETS.items()):
                prog.progress(int(i/len(ABLATION_SETS)*100),text=f"Menguji: {AF.get(name,name)}...")
                Xtr,ytr=tm[fl].values,tm[LABEL_COL].values; Xv,yv=vm[fl].values,vm[LABEL_COL].values; Xts,yts=tsm[fl].values,tsm[LABEL_COL].values
                p=build_rf_pipeline(); p.fit(Xtr,ytr)
                vp=p.predict_proba(Xv)[:,1]; vr=roc_auc_score(yv,vp); vpr=average_precision_score(yv,vp)
                p.fit(np.vstack([Xtr,Xv]),np.concatenate([ytr,yv]))
                tp=p.predict_proba(Xts)[:,1]; tr=roc_auc_score(yts,tp); tpr=average_precision_score(yts,tp)
                ar.append({"Pengujian":AF.get(name,name),"Key":name,"Val ROC":vr,"Val PR":vpr,"Test ROC":tr,"Test PR":tpr})
            prog.progress(100,text="✅ Selesai!")
            adf=pd.DataFrame(ar); fpr=float(adf.loc[adf["Key"]=="FULL","Test PR"].iloc[0]); adf["Penurunan"]=fpr-adf["Test PR"]
            st.dataframe(adf[["Pengujian","Val ROC","Val PR","Test ROC","Test PR","Penurunan"]].style.format({"Val ROC":"{:.4f}","Val PR":"{:.4f}","Test ROC":"{:.4f}","Test PR":"{:.4f}","Penurunan":"{:.4f}"}),use_container_width=True,hide_index=True)
            dd=adf[adf["Key"]!="FULL"].sort_values("Penurunan",ascending=True)
            fig=px.bar(dd,x="Penurunan",y="Pengujian",orientation="h",color="Penurunan",color_continuous_scale=["#1a2332","#ef476f"],template="plotly_dark")
            fig.update_layout(height=300,showlegend=False,paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(17,24,39,0.6)",font=dict(color="#e8ecf1"),xaxis_title="Penurunan Performa (semakin tinggi = semakin penting)",yaxis_title="",title=f"Dampak Penghapusan Fitur (Lengkap = {fpr:.4f})")
            st.plotly_chart(fig,use_container_width=True)

# ── Footer ──
st.markdown("---")
st.markdown("<div style='text-align:center;color:#555;font-size:.85rem;padding:1rem 0'>🌏 <b>AI-based Regional Aftershock Prediction</b> · Physics-Guided Relational Representation (PGRR) · Sumatra · RF · SVM · LR</div>",unsafe_allow_html=True)
