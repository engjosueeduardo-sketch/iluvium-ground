"""
Iluvium Engenharia | Motor Master v3.8
═══════════════════════════════════════════════════════════════════════
Novidades v3.6 (Cores + UX + Curva Média):
  ✓ Sistema unificado de cores por eixo (A=vermelho, B=azul, C=verde...)
     - Tabelas de SEV com banner colorido + abas com sublinhado da cor
     - Mapa: cor segue o eixo selecionado
     - Curva ρ vs profundidade: cada eixo na sua cor
  ✓ Selectbox "Eixo Ativo para Desenhar" — cor sai correta na hora
  ✓ Painel "Editar Desenhos" sempre visível (toggle)
  ✓ Pseudo-Seção 2D substituída por:
     - Curva Média 1D (ρ vs profundidade) com banda min-máx
     - Marcador único representativo no mapa (cor por faixa de ρ)
     - Tabela resumo por profundidade
  ✓ Removidas paleta Res2DInv, máscara cone-de-influência
     (não usadas pela nova abordagem 1D)

Novidades v3.5:
  ✓ Removido Direcionamento de Campo (Azimute) — mapa limpo
  ✓ Painel de edição de desenhos por feature
  ✓ Legenda visual no mapa (rótulo + tooltip)
  ✓ Seção VI no PDF: Mapa de Localização

Novidades v3.4:
  ✓ Draw plugin Folium + persistência GeoJSON

Novidades v3.3:
  ✓ Sistema completo de gestão de laudos (CRUD)
  ✓ Auto-save a cada 30s

Motor geofísico (v3.0+):
  ✓ Coeficiente 2a corrigido
  ✓ Kernel recursivo Stefanescu vetorizado
  ✓ Suporte a 4 camadas
═══════════════════════════════════════════════════════════════════════
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import time, math, tempfile, os
from datetime import datetime
from functools import lru_cache

# Compatibilidade NumPy ≥2.0 (trapezoid) vs <2.0 (trapz)
_trapz = getattr(np, 'trapezoid', None) or getattr(np, 'trapz')

# ── Nomes de colunas em constantes ASCII-safe (evita KeyError com Ω no Py3.8) ──
COL_A   = "Espacamento_m"      # Espaçamento [m]
COL_P   = "Profundidade_m"     # Profundidade [m]
COL_R   = "Resistencia_ohm"    # Resistência [Ω]
COL_RHO = "Resistividade_ohm_m"# Resistividade [Ω·m]

# Labels de exibição (usados no data_editor)
LBL_A   = "Espaçamento [m]"
LBL_P   = "Profundidade [m]"
LBL_R   = "Resistência [Ω]"
LBL_RHO = "Resistividade [Ω·m]"

# =========================================================================
# CARREGAMENTO LAZY
# =========================================================================
@st.cache_resource
def load_scipy():
    import warnings, scipy.integrate as si, scipy.special as ss
    from scipy.optimize import differential_evolution
    warnings.filterwarnings("ignore", category=si.IntegrationWarning)
    return si, ss, differential_evolution

# =========================================================================
# GESTÃO DE ESTADO
# =========================================================================
def reseta_calculo():
    st.session_state.calc_concluido = False
    for k in ['res_x','erro_rms_log','erro_rms_pct','rho_final_calc',
              'camadas_atuais','a_g','rho_g','dir_g','fig_resultado',
              'df_r_resultado','df_d_resultado']:
        st.session_state.pop(k, None)

for k, v in {"calc_concluido": False, "map_zoom": 15,
             "dados_tabelas": {}, "precisa_atualizar": False,
             "chat_historico": [], "chat_api_key": ""}.items():
    if k not in st.session_state:
        st.session_state[k] = v

def check_diff(v1, v2):
    try:
        if pd.isna(v1): v1 = None
        if pd.isna(v2): v2 = None
    except Exception:
        pass
    if v1 is None and v2 is None: return False
    if v1 is None or v2 is None:  return True
    return abs(float(v1) - float(v2)) > 1e-9

# =========================================================================
# UTILITÁRIOS
# =========================================================================
def parse_br_float(val):
    if val is None: return None
    try:
        if isinstance(val, float) and np.isnan(val): return None
    except Exception:
        pass
    try:
        s = str(val).strip()
        return None if s == "" else float(s.replace(',', '.'))
    except Exception:
        return None

def df_novo_eixo():
    """Cria DataFrame padrão com colunas ASCII-safe."""
    return pd.DataFrame({
        COL_A:   [1.0, 2.0, 4.0, 8.0, 16.0, 32.0],
        COL_P:   [0.4] * 6,
        COL_R:   [None] * 6,
        COL_RHO: [400.0, 600.0, 200.0, 100.0, None, None]
    })

def df_para_exibicao(df):
    """Renomeia colunas internas → labels bonitos para o data_editor."""
    return df.rename(columns={
        COL_A: LBL_A, COL_P: LBL_P,
        COL_R: LBL_R, COL_RHO: LBL_RHO})

def df_de_exibicao(df):
    """Renomeia labels do data_editor → colunas internas ASCII-safe."""
    return df.rename(columns={
        LBL_A: COL_A, LBL_P: COL_P,
        LBL_R: COL_R, LBL_RHO: COL_RHO})

def calcular_coordenada_destino(lat1, lon1, az_graus, dist_m):
    R = 6371000.0
    lr, lonr = math.radians(lat1), math.radians(lon1)
    azr = math.radians(az_graus)
    lat2 = math.asin(math.sin(lr)*math.cos(dist_m/R) +
                     math.cos(lr)*math.sin(dist_m/R)*math.cos(azr))
    lon2 = lonr + math.atan2(math.sin(azr)*math.sin(dist_m/R)*math.cos(lr),
                              math.cos(dist_m/R) - math.sin(lr)*math.sin(lat2))
    return math.degrees(lat2), math.degrees(lon2)

# =========================================================================
# UTILS PARA DESENHOS GEOJSON (v3.5)
# =========================================================================
# Cores padrão por eixo (igual COR_MAP mas em hex)
COR_EIXO_HEX = {"A": "#e74c3c", "B": "#3498db", "C": "#2ecc71",
                "D": "#f39c12", "E": "#9b59b6", "F": "#1abc9c",
                "G": "#e67e22", "H": "#34495e"}

def _haversine_m(lat1, lon1, lat2, lon2):
    """Distância em metros entre 2 pontos lat/lon (fórmula de haversine)."""
    R = 6371000.0
    lat1_r = math.radians(lat1); lat2_r = math.radians(lat2)
    dlat = math.radians(lat2 - lat1); dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat/2)**2 +
         math.cos(lat1_r) * math.cos(lat2_r) * math.sin(dlon/2)**2)
    return 2 * R * math.asin(math.sqrt(a))

def comprimento_feature_m(feat):
    """Comprimento de uma LineString GeoJSON em metros. Polygons e Points → 0."""
    geom = feat.get("geometry", {})
    if geom.get("type") not in ("LineString", "Polygon"):
        return 0.0
    coords = geom.get("coordinates", [])
    if geom["type"] == "Polygon":
        coords = coords[0] if coords else []
    if len(coords) < 2:
        return 0.0
    total = 0.0
    for i in range(len(coords) - 1):
        ln1, lt1 = coords[i][:2]
        ln2, lt2 = coords[i+1][:2]
        total += _haversine_m(lt1, ln1, lt2, ln2)
    return total

def descrever_feature(feat):
    """Retorna descrição curta do tipo geométrico."""
    g = feat.get("geometry", {}).get("type", "?")
    return {"LineString":"📏 Linha","Polygon":"🔷 Polígono",
            "Point":"📍 Marcador","Rectangle":"⬜ Retângulo",
            "Circle":"⭕ Círculo"}.get(g, f"❓ {g}")

# =========================================================================
# ███ MOTOR GEOFÍSICO v3.0 (CORRIGIDO) ███
#
# ρ_a(a) = ρ₁ + 2a ∫[T(λ)−ρ₁][J₀(λa)−J₀(2λa)]dλ   ← coef. 2a (era 4a!)
# T(λ): kernel recursivo Stefanescu para N camadas
# =========================================================================
def _kernel_vec(lams, rho_list, h_list):
    T = np.full_like(lams, float(rho_list[-1]))
    for i in range(len(h_list)-1, -1, -1):
        th = np.tanh(np.minimum(lams * float(h_list[i]), 50.0))
        ri = float(rho_list[i])
        T  = ri * (T + ri * th) / (ri + T * th)
    return T

def _rho_a_fast_2d(a_vals, rho_list, h_list, n_lam=400):
    """Motor rápido vetorizado 2D — mudança de variável t=ln(λ)."""
    from scipy.special import j0 as J0
    a_arr = np.atleast_1d(np.array(a_vals, dtype=float))
    a_arr = np.where(a_arr <= 0, 1e-9, a_arr)
    rho1  = float(rho_list[0])
    h_tot = max(sum(float(h) for h in h_list), 0.5)
    lam_min = min(5e-4 / float(np.max(a_arr)),  0.05 / h_tot)
    lam_max = 300.0 / float(np.min(a_arr))
    lams = np.logspace(np.log10(lam_min), np.log10(lam_max), n_lam)
    T    = _kernel_vec(lams, rho_list, h_list)
    la   = lams[:, None] * a_arr[None, :]
    Jd   = J0(la) - J0(2.0 * la)
    intg = (T[:, None] - rho1) * Jd * lams[:, None]
    vals = _trapz(intg, np.log(lams), axis=0)
    return rho1 + 2.0 * a_arr * vals

def _rho_a_sunde2(a_vals, r1, r2, h1, n=120):
    """Sunde (1949) — 2 camadas. Exato e ultra-rápido."""
    a  = np.atleast_1d(np.array(a_vals, dtype=float))
    a  = np.where(a <= 0, 1e-9, a)
    k  = (r2 - r1) / (r2 + r1)
    ns = np.arange(1, n+1)[:, None]
    t1 = 1.0 / np.sqrt(1.0 + (2.0 * ns * h1 / a[None,:]) ** 2)
    t2 = 1.0 / np.sqrt(4.0 + (2.0 * ns * h1 / a[None,:]) ** 2)
    return r1 * (1.0 + 4.0 * np.sum((k ** ns) * (t1 - t2), axis=0))

def modelo_fast(a_vals, params, nc):
    p = params; a = np.atleast_1d(np.array(a_vals, dtype=float))
    if   nc == 2: return _rho_a_sunde2(a, p[0], p[1], p[2])
    elif nc == 3: return _rho_a_fast_2d(a, [p[0],p[1],p[2]], [p[3],p[4]])
    elif nc == 4: return _rho_a_fast_2d(a, [p[0],p[1],p[2],p[3]], [p[4],p[5],p[6]])
    return np.full_like(a, np.nan)

@lru_cache(maxsize=6000)
def _hankel_cached(a_r, rho_t, h_t):
    from scipy.integrate import quad
    from scipy.special import j0 as J0
    rl, hl = list(rho_t), list(h_t)
    rho1 = rl[0]; a = float(a_r)
    def integrand(lam):
        T = float(_kernel_vec(np.array([lam]), rl, hl)[0])
        return (T - rho1) * (J0(lam*a) - J0(2.0*lam*a))
    val, _ = quad(integrand, 1e-7, 300.0/a, limit=400, epsabs=1e-12, epsrel=1e-10)
    return rho1 + 2.0 * a * val

def _p2t(params, nc):
    p = params
    if   nc == 2: return (float(p[0]),float(p[1])), (float(p[2]),)
    elif nc == 3: return ((float(p[0]),float(p[1]),float(p[2])),
                          (float(p[3]),float(p[4])))
    elif nc == 4: return ((float(p[0]),float(p[1]),float(p[2]),float(p[3])),
                          (float(p[4]),float(p[5]),float(p[6])))
    return tuple(float(x) for x in p[:nc]), tuple(float(x) for x in p[nc:])

def modelo_exact(a_vals, params, nc):
    rho_t, h_t = _p2t(params, nc)
    return np.array([_hankel_cached(round(float(a), 7), rho_t, h_t)
                     for a in np.atleast_1d(np.array(a_vals, dtype=float))])

# =========================================================================
# CORREÇÃO DE HASTE (PALMER)
# =========================================================================
def palmer_rho_aparente(a, r, p, aplicar=True):
    if not aplicar or p <= 0: return 2.0 * math.pi * a * r
    t1 = (2*a) / math.sqrt(a**2 + 4*p**2)
    t2 =    a  / math.sqrt(a**2 +   p**2)
    return (4.0 * math.pi * a * r) / (1.0 + t1 - t2)

def palmer_resistencia(a, rho, p, aplicar=True):
    if not aplicar or p <= 0: return rho / (2.0 * math.pi * a)
    t1 = (2*a) / math.sqrt(a**2 + 4*p**2)
    t2 =    a  / math.sqrt(a**2 +   p**2)
    return (rho * (1.0 + t1 - t2)) / (4.0 * math.pi * a)

# =========================================================================
# LAUDO PDF
# =========================================================================
def renderiza_mapa_estatico_png(lat, lon, desenhos_geojson, largura=1100, altura=700):
    """Renderiza mapa satélite com desenhos. Auto-zoom focado nos features desenhados."""
    try:
        from staticmap import StaticMap, Line, CircleMarker, Polygon
        from PIL import Image, ImageDraw, ImageFont
        import io, math as _m
    except ImportError as e:
        return None, f"Bibliotecas faltando: {e}"

    try:
        TILE_URL = "https://tile.openstreetmap.org/{z}/{x}/{y}.png"

        # Coleta APENAS coordenadas dos desenhos (não inclui o centro GPS)
        # → o zoom foca exatamente onde o usuário desenhou
        feat_coords = []
        for feat in (desenhos_geojson or {}).get("features", []):
            geom = feat.get("geometry", {})
            gtype = geom.get("type", "")
            coords = geom.get("coordinates", [])
            if gtype == "Point" and len(coords) >= 2:
                feat_coords.append((coords[0], coords[1]))
            elif gtype == "LineString":
                feat_coords.extend([(c[0], c[1]) for c in coords if len(c) >= 2])
            elif gtype == "Polygon":
                ring = coords[0] if coords else []
                feat_coords.extend([(c[0], c[1]) for c in ring if len(c) >= 2])

        if feat_coords:
            # Centro = centroide dos desenhos
            centro_lon = sum(c[0] for c in feat_coords) / len(feat_coords)
            centro_lat = sum(c[1] for c in feat_coords) / len(feat_coords)
            min_lon = min(c[0] for c in feat_coords)
            max_lon = max(c[0] for c in feat_coords)
            min_lat = min(c[1] for c in feat_coords)
            max_lat = max(c[1] for c in feat_coords)
            # Margem de 20% ao redor dos desenhos
            margem = 0.20
            span_lon = max(max_lon - min_lon, 0.0005)
            span_lat = max(max_lat - min_lat, 0.0005)
            min_lon -= span_lon * margem; max_lon += span_lon * margem
            min_lat -= span_lat * margem; max_lat += span_lat * margem
            spread = max(max_lon - min_lon, max_lat - min_lat)
            # Zoom calculado pelo spread com margem
            if   spread < 0.001:  zoom = 19
            elif spread < 0.003:  zoom = 18
            elif spread < 0.007:  zoom = 17
            elif spread < 0.015:  zoom = 16
            elif spread < 0.03:   zoom = 15
            elif spread < 0.07:   zoom = 14
            elif spread < 0.15:   zoom = 13
            else:                 zoom = 12
        else:
            # Sem desenhos → usa o centro GPS com zoom padrão
            centro_lon, centro_lat = lon, lat
            zoom = 17

        m = StaticMap(largura, altura, url_template=TILE_URL)

        # Marcador do centro GPS (ponto amarelo pequeno)
        m.add_marker(CircleMarker((lon, lat), 'yellow', 10))
        m.add_marker(CircleMarker((lon, lat), '#000000', 4))

        # Adiciona cada feature na cor do eixo
        legenda_items = []  # (eixo, cor_hex, comprimento_total, qtd)
        resumo_eixos = {}

        for feat in (desenhos_geojson or {}).get("features", []):
            props = feat.get("properties", {})
            eixo = props.get("eixo", "—")
            cor = COR_EIXO_HEX.get(eixo, props.get("color", "#3388ff"))
            geom = feat.get("geometry", {})
            gtype = geom.get("type", "")
            coords = geom.get("coordinates", [])
            comp = comprimento_feature_m(feat)

            try:
                if gtype == "Point" and len(coords) >= 2:
                    m.add_marker(CircleMarker((coords[0], coords[1]), cor, 12))
                elif gtype == "LineString":
                    pts = [(c[0], c[1]) for c in coords if len(c) >= 2]
                    if len(pts) >= 2:
                        m.add_line(Line(pts, cor, 6))
                elif gtype == "Polygon":
                    ring = coords[0] if coords else []
                    pts = [(c[0], c[1]) for c in ring if len(c) >= 2]
                    if len(pts) >= 3:
                        m.add_polygon(Polygon(pts, cor + "80", cor, simplify=False))
            except Exception:
                continue

            if eixo != "—":
                resumo_eixos.setdefault(eixo, {"qtd":0, "comp":0.0, "cor":cor})
                resumo_eixos[eixo]["qtd"] += 1
                resumo_eixos[eixo]["comp"] += comp

        img = m.render(zoom=zoom, center=[centro_lon, centro_lat])

        # Adiciona LEGENDA sobreposta no canto inferior direito
        if resumo_eixos:
            draw = ImageDraw.Draw(img, "RGBA")
            try:
                font = ImageFont.truetype("DejaVuSans-Bold.ttf", 16)
                font_small = ImageFont.truetype("DejaVuSans.ttf", 13)
            except Exception:
                try:
                    from PIL import ImageFont as IF
                    font = IF.load_default(); font_small = IF.load_default()
                except Exception:
                    font = font_small = None

            n_items = len(resumo_eixos) + 1
            box_h = 40 + n_items * 22
            box_w = 240
            margin = 12
            x0 = largura - box_w - margin
            y0 = altura - box_h - margin
            # Fundo semitransparente
            draw.rectangle([x0, y0, x0+box_w, y0+box_h],
                          fill=(255,255,255,230), outline=(20,20,20,255), width=2)
            # Título
            if font:
                draw.text((x0+10, y0+8), "LEGENDA", fill=(20,20,20), font=font)
            # Linha separadora
            draw.line([x0+10, y0+30, x0+box_w-10, y0+30], fill=(80,80,80), width=1)
            # Items
            y = y0 + 38
            for eixo_x, dados in sorted(resumo_eixos.items()):
                # Quadrado de cor
                cor_hex = dados["cor"].lstrip("#")
                rgb = tuple(int(cor_hex[i:i+2], 16) for i in (0,2,4))
                draw.rectangle([x0+12, y+3, x0+30, y+18], fill=rgb,
                              outline=(20,20,20), width=1)
                # Texto
                if font_small:
                    txt = f"Eixo {eixo_x}  ·  {dados['comp']:.0f} m"
                    draw.text((x0+38, y+1), txt, fill=(20,20,20), font=font_small)
                y += 22

        buf = io.BytesIO()
        img.save(buf, format='PNG')
        return buf.getvalue(), None
    except Exception as e:
        return None, f"Erro ao gerar mapa: {e}"

def gerar_laudo_pdf(df_res, df_desv, rms_log, rms_pct, lat, lon,
                    fig, cliente, projeto, obs, tecnico, crea,
                    fig_ps=None, desenhos_geojson=None):
    from fpdf import FPDF

    class LP(FPDF):
        def header(self):
            self.set_font('Arial','B',16); self.set_text_color(25,25,112)
            self.cell(0,10,'ILUVIUM ENGENHARIA',ln=True,align='C')
            self.set_font('Arial','B',11); self.set_text_color(80,80,80)
            self.cell(0,6,'LAUDO DE ESTRATIFICACAO DO SOLO - NBR 7117',ln=True,align='C')
            self.line(10,30,200,30); self.ln(8)
        def footer(self):
            self.set_y(-15); self.set_font('Arial','I',8); self.set_text_color(128)
            self.cell(0,10,f'Pagina {self.page_no()} | Iluvium v3.1',align='C')

    def s(t): return str(t).encode('latin-1','replace').decode('latin-1')

    pdf = LP(); pdf.add_page()
    for lbl, val in [("Cliente / Empresa:", cliente or "Nao informado"),
                     ("Projeto / Obra:", projeto or "Malha de Aterramento"),
                     ("Data:", datetime.now().strftime('%d/%m/%Y %H:%M'))]:
        pdf.set_font('Arial','B',11); pdf.cell(55,7,s(lbl))
        pdf.set_font('Arial','',11);  pdf.cell(0,7,s(val),ln=True)
    pdf.ln(3)
    pdf.set_font('Arial','B',12); pdf.set_fill_color(230,230,250)
    pdf.cell(0,9,'I. RESUMO TECNICO GEOFISICO',ln=True,fill=True)
    pdf.set_font('Arial','',10); pdf.ln(2)
    pdf.multi_cell(0,6,
        f"Coordenadas: Lat {lat:.6f} / Lon {lon:.6f}\n"
        f"Erro RMS log: {rms_log:.4f}  |  Erro RMS (%): {rms_pct:.2f}%\n"
        "Motor: Kernel Stefanescu recursivo | Coef. 2a corrigido (v3.1)")
    pdf.ln(3)
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    tmp.close()  # Windows: precisa fechar antes de outro processo usar
    try:
        fig.write_image(tmp.name, engine="kaleido", width=1100, height=520, scale=2)
        pdf.image(tmp.name, x=8, w=194)
    finally:
        try: os.remove(tmp.name)
        except (OSError, PermissionError): pass

    pdf.add_page()
    pdf.set_font('Arial','B',12); pdf.set_fill_color(210,210,210)
    pdf.cell(0,10,'II. MODELO DE SOLO (PROJETO)',ln=True,fill=True)
    pdf.set_font('Arial','B',10)
    for hd,w in [('Camada',55),('Resistividade (Ohm.m)',70),('Espessura (m)',65)]:
        pdf.cell(w,8,s(hd),border=1,fill=True,align='C')
    pdf.ln(); pdf.set_font('Arial','',10)
    for _,row in df_res.iterrows():
        nm = s(str(row.get('Camada','')))
        pdf.cell(55,8,nm,border=1,align='C')
        pdf.cell(70,8,s(str(row.get('rho',''))),border=1,align='C')
        pdf.cell(65,8,s(str(row.get('h',''))),border=1,align='C')
        pdf.ln()
    pdf.ln(5); pdf.set_font('Arial','B',12)
    pdf.cell(0,10,'III. MEMORIA DE CALCULO',ln=True,fill=True)
    pdf.set_font('Arial','B',10)
    for hd,w in [('a (m)',38),('Eixo',22),('Medida',44),('Calculada',48),('Desvio',38)]:
        pdf.cell(w,8,s(hd),border=1,fill=True,align='C')
    pdf.ln(); pdf.set_font('Arial','',10)
    for _,row in df_desv.iterrows():
        pdf.cell(38,8,s(str(row.get('a [m]',''))),border=1,align='C')
        pdf.cell(22,8,s(str(row.get('Eixo',''))),border=1,align='C')
        pdf.cell(44,8,s(str(row.get('Medida',''))),border=1,align='C')
        pdf.cell(48,8,s(str(row.get('Calculada',''))),border=1,align='C')
        dv = float(row.get('Desvio %',0))
        if abs(dv)>20: pdf.set_text_color(200,20,20)
        elif abs(dv)>10: pdf.set_text_color(200,100,0)
        pdf.cell(38,8,s(f"{dv:.2f}%"),border=1,align='C')
        pdf.set_text_color(0,0,0); pdf.ln()
    if obs:
        pdf.ln(5); pdf.set_font('Arial','B',12)
        pdf.cell(0,10,'IV. PARECER TECNICO',ln=True,fill=True)
        pdf.set_font('Arial','',10); pdf.multi_cell(0,6,s(obs))

    # ── V. SEÇÃO GEOELÉTRICA 2D (estilo Res2DInv) ─────────────────────────
    if fig_ps is not None:
        pdf.add_page()
        pdf.set_font('Arial','B',12); pdf.set_fill_color(210,210,210)
        pdf.cell(0,10,'V. SECAO GEOELETRICA 2D (TOMOGRAFIA DE RESISTIVIDADE)',
                 ln=True, fill=True)
        pdf.set_font('Arial','',10); pdf.ln(3)
        pdf.multi_cell(0,6, s(
            "A seção 2D abaixo apresenta a distribuicao espacial da resistividade "
            "aparente do solo, gerada por interpolacao cubica sobre os ensaios SEV "
            "executados nos eixos cadastrados. A escala cromatica e logaritmica em "
            "16 niveis (paleta padrao Res2DInv). A mascara trapezoidal corresponde "
            "ao volume efetivamente sondado pelos arranjos Wenner."))
        pdf.ln(2)
        try:
            tmp2 = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
            tmp2.close()
            try:
                fig_ps.write_image(tmp2.name, engine="kaleido",
                                   width=1400, height=600, scale=2)
                pdf.image(tmp2.name, x=8, w=194)
            finally:
                try: os.remove(tmp2.name)
                except (OSError, PermissionError): pass
        except Exception as e:
            pdf.set_text_color(180,0,0)
            pdf.multi_cell(0,6, s(f"[Falha ao renderizar figura: {e}]"))
            pdf.set_text_color(0,0,0)
        pdf.ln(2)
        pdf.set_font('Arial','I',9); pdf.set_text_color(60,60,60)
        legenda = (f"Figura 1 - Secao geoeletrica 2D obtida pelo metodo de Wenner. "
                   f"RMS = {rms_pct:.2f}%. Fonte: Iluvium Engenharia, "
                   f"{datetime.now().year}.")
        pdf.multi_cell(0,5, s(legenda), align='C')
        pdf.set_text_color(0,0,0)

    # ── V.b RESISTIVIDADE MÉDIA vs PROFUNDIDADE (Wenner) ──────────────────
    if df_res is not None:
        try:
            # Reconstrói dados a partir dos resultados disponíveis
            _dados_pdf = {}
            for _, row in df_res.iterrows():
                _eixo = str(row.get("Eixo", "A"))
                _a    = float(row.get("Medida", 0)) if row.get("Medida") else None
                _rho  = float(row.get("Calculada", 0)) if row.get("Calculada") else None
                if _a and _rho:
                    _dados_pdf.setdefault(_eixo, []).append((round(0.519*_a, 2), _rho))

            if _dados_pdf:
                pdf.add_page()
                pdf.set_font('Arial','B',12); pdf.set_fill_color(210,210,210)
                pdf.cell(0,10,'V.b RESISTIVIDADE MEDIA vs PROFUNDIDADE (WENNER)',
                         ln=True, fill=True)
                pdf.set_font('Arial','',10); pdf.ln(3)
                pdf.multi_cell(0,6, s(
                    "Curva de resistividade aparente media por profundidade efetiva "
                    "de investigacao (z = 0.519 x a, Roy & Apparao, 1971), "
                    "calculada sobre todos os eixos do ensaio. "
                    "A faixa cinza indica a variabilidade min-max entre eixos."))
                pdf.ln(3)

                # Gráfico via kaleido
                try:
                    import plotly.graph_objects as _go
                    _todos = [(z, r) for pares in _dados_pdf.values() for (z, r) in pares]
                    _z_u   = sorted(set(round(z,2) for z,_ in _todos))
                    _curva = []
                    for _z in _z_u:
                        _vs = [r for z,r in _todos if abs(z-_z)<=max(0.05*_z,0.01)]
                        if _vs:
                            _curva.append({"z": _z, "med": float(sum(_vs)/len(_vs)),
                                           "mn": min(_vs), "mx": max(_vs)})
                    if _curva:
                        _fig_pdf = _go.Figure()
                        _za  = [p["z"]   for p in _curva]
                        _rme = [p["med"] for p in _curva]
                        _rmi = [p["mn"]  for p in _curva]
                        _rma = [p["mx"]  for p in _curva]
                        _fig_pdf.add_trace(_go.Scatter(
                            x=_rma+_rmi[::-1], y=_za+_za[::-1],
                            fill='toself', fillcolor='rgba(150,150,150,0.25)',
                            line=dict(color='rgba(0,0,0,0)'),
                            showlegend=True, name='Faixa min/máx'))
                        for _d, _pares in _dados_pdf.items():
                            _zd = [p[0] for p in _pares]; _rd = [p[1] for p in _pares]
                            _fig_pdf.add_trace(_go.Scatter(
                                x=_rd, y=_zd, mode='lines+markers',
                                name=f'Eixo {_d}',
                                line=dict(width=2, dash='dot')))
                        _fig_pdf.add_trace(_go.Scatter(
                            x=_rme, y=_za, mode='lines+markers',
                            name='<b>Média</b>',
                            line=dict(color='#0a0a0a', width=3),
                            marker=dict(size=10, color='#c0392b', symbol='diamond')))
                        _fig_pdf.update_layout(
                            xaxis=dict(title='ρ (Ω·m)', type='log'),
                            yaxis=dict(title='Profundidade z (m)', autorange='reversed'),
                            template='plotly_white', height=400, width=900,
                            margin=dict(t=30, b=50, l=70, r=20),
                            legend=dict(x=0.98, y=0.02, xanchor='right'))
                        _tmpg = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
                        _tmpg.close()
                        try:
                            _fig_pdf.write_image(_tmpg.name, engine='kaleido',
                                                 width=900, height=400, scale=2)
                            pdf.image(_tmpg.name, x=10, w=190)
                        finally:
                            try: os.remove(_tmpg.name)
                            except (OSError, PermissionError): pass

                    # Tabela numérica
                    pdf.ln(4)
                    pdf.set_font('Arial','B',10)
                    pdf.cell(0,7,s('Tabela de Resistividade Media por Profundidade:'), ln=True)
                    pdf.set_font('Arial','',9)
                    pdf.set_fill_color(220,220,220)
                    for _h, _w in zip(['Prof. z (m)','Espaç. a (m)','ρ médio (Ω·m)',
                                       'ρ mín','ρ máx','Eixos'],[30,30,38,28,28,26]):
                        pdf.cell(_w,7,s(_h), border=1, fill=True, align='C')
                    pdf.ln()
                    pdf.set_fill_color(255,255,255)
                    for _p in (_curva if _curva else []):
                        _a_val = round(_p['z']/0.519, 2)
                        _n     = len([r for z,r in _todos if abs(z-_p['z'])<=max(0.05*_p['z'],0.01)])
                        for _txt, _w in zip([f"{_p['z']:.2f}", f"{_a_val:.2f}",
                                              f"{_p['med']:.1f}", f"{_p['mn']:.1f}",
                                              f"{_p['mx']:.1f}", str(_n)],
                                            [30,30,38,28,28,26]):
                            pdf.cell(_w,6,s(_txt), border=1, align='C')
                        pdf.ln()
                    # Nota de rodapé
                    pdf.ln(3)
                    pdf.set_font('Arial','I',9); pdf.set_text_color(60,60,60)
                    pdf.multi_cell(0,5,s(
                        "z = 0.519 x a (Roy & Apparao, 1971). "
                        "Metodo Wenner. Resistividade aparente medida em campo."),
                        align='C')
                    pdf.set_text_color(0,0,0)
                except Exception as _eg:
                    pdf.set_text_color(150,0,0)
                    pdf.multi_cell(0,6,s(f'[Grafico nao gerado: {_eg}]'))
                    pdf.set_text_color(0,0,0)
        except Exception:
            pass

    # ── V.c COLUNA GEOELÉTRICA (log de sondagem estilo NBR) ───────────────
    _fig_col_pdf = st.session_state.get('fig_coluna') if hasattr(st, 'session_state') else None
    # Na geração do PDF, reconstruímos a coluna diretamente dos dados
    if df_res is not None:
        try:
            import math as _mth, io as _io
            _PALETA_PDF = [
                (8,   0,107),(13,  0,201),(0,  23,255),(0, 102,255),
                (0, 180,255),(0, 255,255),(127,255,212),(0, 250,154),
                (0, 255,  0),(154,205, 50),(255,255,  0),(255,199,  0),
                (255,140,  0),(255, 69,  0),(255,  0,  0),(139,  0,  0),
            ]
            # Reconstrói a curva média a partir dos resultados
            _dados_col = {}
            for _, row in df_res.iterrows():
                _a2   = float(row.get("Medida", 0) or 0)
                _rho2 = float(row.get("Calculada", 0) or 0)
                if _a2 and _rho2:
                    _z2 = round(0.519 * _a2, 2)
                    _dados_col.setdefault(_z2, []).append(_rho2)

            if _dados_col:
                _curva_pdf = sorted([
                    {"z": z, "med": sum(vs)/len(vs)} for z, vs in _dados_col.items()
                ], key=lambda x: x["z"])

                _rlo = _mth.log10(max(min(p["med"] for p in _curva_pdf), 1))
                _rhi = _mth.log10(max(max(p["med"] for p in _curva_pdf), _rlo + 0.01))

                def _hex_pdf(rho):
                    t = (_mth.log10(max(rho,1))-_rlo)/(_rhi-_rlo)
                    t = max(0., min(1., t))
                    r,g,b = _PALETA_PDF[min(int(t*16),15)]
                    return f"#{r:02X}{g:02X}{b:02X}"

                def _tc_pdf(rho):
                    t = (_mth.log10(max(rho,1))-_rlo)/(_rhi-_rlo)
                    t = max(0.,min(1.,t))
                    ri,gi,bi = _PALETA_PDF[min(int(t*16),15)]
                    return 'black' if 0.299*ri+0.587*gi+0.114*bi > 140 else 'white'

                _fig_vc = go.Figure()
                for _ki, _pp in enumerate(_curva_pdf):
                    _zt = 0.0 if _ki==0 else _curva_pdf[_ki-1]["z"]
                    _zb = _pp["z"]
                    _fig_vc.add_shape(
                        type="rect", x0=0, x1=1, y0=_zt, y1=_zb,
                        fillcolor=_hex_pdf(_pp["med"]),
                        line=dict(color="black", width=1.2), layer="below")
                    _fig_vc.add_annotation(
                        x=0.5, y=(_zt+_zb)/2,
                        text=f"<b>{_pp['med']:.0f} Ω·m</b>",
                        showarrow=False,
                        font=dict(size=13, color=_tc_pdf(_pp["med"]), family="Arial"),
                        xanchor="center", yanchor="middle")
                # Barra de cores horizontal
                _zm = max(p["z"] for p in _curva_pdf)
                for _ci,(_ri,_gi,_bi) in enumerate(_PALETA_PDF):
                    _fig_vc.add_shape(
                        type="rect",
                        x0=_ci/16, x1=(_ci+1)/16, y0=_zm+0.15, y1=_zm+0.55,
                        fillcolor=f"#{_ri:02X}{_gi:02X}{_bi:02X}",
                        line=dict(color="black", width=0.5), layer="below")
                _rho_min_str = f"{min(p['med'] for p in _curva_pdf):.0f}"
                _rho_max_str = f"{max(p['med'] for p in _curva_pdf):.0f}"
                _fig_vc.add_annotation(x=0, y=_zm+0.7,
                    text=f"<b>{_rho_min_str}</b>",
                    showarrow=False, font=dict(size=9), xanchor="left")
                _fig_vc.add_annotation(x=0.5, y=_zm+0.7,
                    text="<b>Resistividade (Ω·m)</b>",
                    showarrow=False, font=dict(size=9), xanchor="center")
                _fig_vc.add_annotation(x=1, y=_zm+0.7,
                    text=f"<b>{_rho_max_str}</b>",
                    showarrow=False, font=dict(size=9), xanchor="right")
                _fig_vc.update_layout(
                    xaxis=dict(visible=False, range=[0,1]),
                    yaxis=dict(title="Profundidade z (m)", autorange="reversed",
                               showgrid=True, gridcolor="#ddd",
                               tickfont=dict(size=11, family="Arial")),
                    template="plotly_white", height=500, width=400,
                    margin=dict(t=20, b=10, l=60, r=10),
                    plot_bgcolor="white", paper_bgcolor="white")

                pdf.add_page()
                pdf.set_font('Arial','B',12); pdf.set_fill_color(210,210,210)
                pdf.cell(0,10,'V.c COLUNA GEOELETRICA (LOG DE SONDAGEM)',
                         ln=True, fill=True)
                pdf.set_font('Arial','',10); pdf.ln(3)
                pdf.multi_cell(0,6, s(
                    "Coluna geoeletrica vertical representando as faixas de resistividade "
                    "por profundidade efetiva de investigacao (z = 0.519 x a, Wenner). "
                    "Paleta cromatica conforme NBR 7117 / Res2DInv: azul = baixa "
                    "resistividade, vermelho = alta resistividade."))
                pdf.ln(2)
                _tvc = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
                _tvc.close()
                try:
                    _fig_vc.write_image(_tvc.name, engine='kaleido',
                                        width=400, height=520, scale=2)
                    # Centraliza a imagem (largura 80mm, centrada em 210mm)
                    pdf.image(_tvc.name, x=65, w=80)
                finally:
                    try: os.remove(_tvc.name)
                    except (OSError, PermissionError): pass
                pdf.ln(3)
                pdf.set_font('Arial','I',9); pdf.set_text_color(60,60,60)
                pdf.multi_cell(0,5, s(
                    "Figura 3 - Coluna geoeletrica da area ensaiada. "
                    "Cada faixa representa a resistividade media naquela profundidade."),
                    align='C')
                pdf.set_text_color(0,0,0)
        except Exception as _evc:
            pass  # Não quebra o PDF se a coluna falhar

    # ── VI. MAPA DE LOCALIZAÇÃO E EIXOS DA MALHA ───────────────────────
    if desenhos_geojson and len(desenhos_geojson.get("features", [])) > 0:
        pdf.add_page()
        pdf.set_font('Arial','B',12); pdf.set_fill_color(210,210,210)
        pdf.cell(0,10,'VI. MAPA DE LOCALIZACAO E EIXOS DA MALHA',
                 ln=True, fill=True)
        pdf.set_font('Arial','',10); pdf.ln(3)

        n_feat = len(desenhos_geojson["features"])
        pdf.multi_cell(0,6, s(
            f"Mapa satelite (ESRI World Imagery) sobre a area do ensaio com "
            f"{n_feat} elemento(s) georreferenciado(s) sobreposto(s). As linhas "
            f"coloridas e poligonos indicam os eixos SEV e a area da malha de "
            f"aterramento conforme cadastrado pelo engenheiro responsavel."))
        pdf.ln(2)

        # Renderiza mapa estático
        png_bytes, err = renderiza_mapa_estatico_png(
            lat, lon, desenhos_geojson, largura=1100, altura=720)

        if png_bytes:
            tmp_map = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
            tmp_map.close()
            try:
                with open(tmp_map.name, 'wb') as fp:
                    fp.write(png_bytes)
                pdf.image(tmp_map.name, x=10, w=190)
            finally:
                try: os.remove(tmp_map.name)
                except (OSError, PermissionError): pass

            # Legenda da figura
            pdf.ln(2)
            pdf.set_font('Arial','I',9); pdf.set_text_color(60,60,60)
            pdf.multi_cell(0,5, s(
                f"Figura 2 - Mapa de localizacao da area do ensaio "
                f"({lat:.6f}, {lon:.6f}). Desenhos coloridos: eixos da "
                f"malha cadastrados via plataforma Iluvium."), align='C')
            pdf.set_text_color(0,0,0)

            # Tabela detalhada de cada feature
            pdf.ln(4)
            pdf.set_font('Arial','B',10)
            pdf.cell(0,7,s("Detalhamento dos elementos georreferenciados:"), ln=True)
            pdf.set_font('Arial','',9)
            pdf.set_fill_color(220,220,220)

            # Cabeçalho
            cab_w = [10, 35, 25, 28, 92]
            for txt, w in zip(["#","Tipo","Eixo","Compr.(m)","Observação"], cab_w):
                pdf.cell(w, 7, s(txt), border=1, fill=True, align='C')
            pdf.ln()

            # Linhas
            for i, feat in enumerate(desenhos_geojson["features"], 1):
                props = feat.get("properties", {})
                tipo_str = descrever_feature(feat).split(" ", 1)[1] if " " in descrever_feature(feat) else "—"
                eixo_str = props.get("eixo", "—")
                comp_str = f"{comprimento_feature_m(feat):.1f}"
                obs_str = (props.get("obs", "") or "—")[:50]

                # Cor da célula do eixo
                if eixo_str in COR_EIXO_HEX:
                    hex_c = COR_EIXO_HEX[eixo_str].lstrip("#")
                    rgb = tuple(int(hex_c[j:j+2], 16) for j in (0,2,4))
                    pdf.set_fill_color(*rgb)
                    pdf.cell(cab_w[0], 6, s(str(i)), border=1, align='C')
                    pdf.cell(cab_w[1], 6, s(tipo_str), border=1, align='C')
                    pdf.set_text_color(255,255,255)
                    pdf.cell(cab_w[2], 6, s(eixo_str), border=1, align='C', fill=True)
                    pdf.set_text_color(0,0,0)
                    pdf.set_fill_color(255,255,255)
                    pdf.cell(cab_w[3], 6, s(comp_str), border=1, align='R')
                    pdf.cell(cab_w[4], 6, s(obs_str), border=1)
                else:
                    for txt, w in zip([str(i), tipo_str, eixo_str, comp_str, obs_str], cab_w):
                        pdf.cell(w, 6, s(txt), border=1, align='C' if w<40 else 'L')
                pdf.ln()

            # Resumo por eixo
            resumo = {}
            for f in desenhos_geojson["features"]:
                e = f.get("properties", {}).get("eixo", "—")
                if e == "—": continue
                resumo.setdefault(e, {"qtd":0, "comp":0.0})
                resumo[e]["qtd"] += 1
                resumo[e]["comp"] += comprimento_feature_m(f)

            if resumo:
                pdf.ln(4)
                pdf.set_font('Arial','B',10)
                pdf.cell(0,7,s("Resumo por eixo:"), ln=True)
                pdf.set_font('Arial','',9)
                for eixo_x, dados in sorted(resumo.items()):
                    pdf.cell(0, 6, s(
                        f"  Eixo {eixo_x}: {dados['qtd']} item(ns), "
                        f"comprimento total = {dados['comp']:.1f} m"), ln=True)
        else:
            pdf.set_font('Arial','I',9); pdf.set_text_color(180,0,0)
            pdf.multi_cell(0,6, s(
                f"[Mapa nao pode ser renderizado: {err or 'erro desconhecido'}. "
                f"Os desenhos georreferenciados estao disponiveis no arquivo "
                f"GeoJSON exportavel atraves da plataforma Iluvium.]"))
            pdf.set_text_color(0,0,0)

    pdf.ln(18); pdf.set_font('Arial','',10)
    pdf.cell(0,5,'_'*55,ln=True,align='C'); pdf.set_font('Arial','B',10)
    pdf.cell(0,6,s(tecnico),ln=True,align='C')
    if crea: pdf.cell(0,6,s(f"CREA/CFT: {crea}"),ln=True,align='C')
    return bytes(pdf.output())

@st.dialog("📋 Dados Formais do Laudo")
def modal_pdf(df_res, df_desv, rms_log, rms_pct, lat, lon, fig,
              fig_ps=None, desenhos_geojson=None):
    st.info("Preencha os dados para gerar o laudo técnico PDF.")
    if fig_ps is not None:
        st.success("✅ Pseudo-Seção 2D será incluída no laudo (Seção V)")
    else:
        st.warning("⚠️ Pseudo-Seção 2D não disponível — adicione ≥2 eixos para incluí-la.")
    if desenhos_geojson and len(desenhos_geojson.get("features", [])) > 0:
        n_d = len(desenhos_geojson["features"])
        st.success(f"✅ Mapa de localização será incluído no laudo (Seção VI · {n_d} desenho(s))")
    else:
        st.info("ℹ️ Nenhum desenho cadastrado. Para incluir mapa no laudo, "
                "ative '✏️ Modo Desenho' e marque eixos no mapa.")
    c1, c2 = st.columns(2)
    cliente  = c1.text_input("Cliente / Empresa:")
    projeto  = c2.text_input("Título do Projeto:", value="Malha de Aterramento Industrial")
    c3, c4   = st.columns(2)
    tecnico  = c3.text_input("Engenheiro Responsável:", value="Eng. Responsável")
    crea     = c4.text_input("CREA/CFT:", placeholder="Ex: CREA-SP 12345678")
    obs      = st.text_area("Parecer Técnico (Opcional):")
    if st.button("📄 Gerar e Baixar PDF", type="primary", use_container_width=True):
        with st.spinner("Compilando laudo..."):
            try:
                pdf_b = gerar_laudo_pdf(df_res, df_desv, rms_log, rms_pct,
                                        lat, lon, fig, cliente, projeto, obs,
                                        tecnico, crea, fig_ps=fig_ps,
                                        desenhos_geojson=desenhos_geojson)
                st.success("Laudo gerado!")
                st.download_button("⬇️ Baixar Laudo PDF", data=pdf_b,
                    file_name=f"Laudo_Iluvium_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
                    mime="application/pdf", use_container_width=True)
                # Log no laudo aberto
                if st.session_state.get("laudo_atual_id"):
                    adicionar_log(
                        st.session_state.laudo_atual_id,
                        "pdf_gerado",
                        f"Cliente: {cliente} | Projeto: {projeto} | "
                        f"{len(pdf_b)//1024} KB")
                    # Atualiza parecer no estado para próximo save
                    st.session_state.parecer_atual = obs
            except Exception as e:
                st.error("Erro ao gerar PDF."); st.exception(e)

# =========================================================================
# ███ CHAT IA — ASSISTENTE DE ENGENHARIA ███
# =========================================================================
SYSTEM_PROMPT_ENGENHARIA = """Você é um assistente especialista em engenharia elétrica,
com foco em sistemas de aterramento, estratificação de solo e proteção elétrica.
Você conhece profundamente as seguintes normas e referências técnicas:

NORMAS BRASILEIRAS (ABNT):
- NBR 7117: Medição da resistividade do solo — Método Wenner
- NBR 5410: Instalações elétricas de baixa tensão
- NBR 5419: Proteção de estruturas contra descargas atmosféricas (SPDA)
- NBR 14039: Instalações elétricas de média tensão
- NBR IEC 61936-1: Instalações de potência acima de 1 kV
- NBR 10898: Sistema de iluminação de emergência

NORMAS INTERNACIONAIS:
- IEEE Std 80: Guide for Safety in AC Substation Grounding
- IEEE Std 81: Guide for Measuring Earth Resistivity (Wenner, Schlumberger, Dipole-Dipole)
- IEC 60364: Instalações elétricas de edificações
- IEC 60479: Efeitos da corrente elétrica no corpo humano

TEORIA GEOFÍSICA:
- Método Wenner: configuração A–M–N–B, espaçamento uniforme a
  ρ_a = 2πa·R (sem correção) | fórmula de Palmer para correção de haste
- Estratificação multicamadas: método de Stefanescu, kernel recursivo
  T_N = ρ_N; T_n = ρ_n·(T_{n+1}+ρ_n·tanh(λhₙ))/(ρ_n+T_{n+1}·tanh(λhₙ))
- Integral de Wenner correta: ρ_a = ρ₁ + 2a·∫[T(λ)-ρ₁][J₀(λa)-J₀(2λa)]dλ
- Modelos típicos: 2 camadas (homogêneo simples), 3 camadas (NBR padrão),
  4 camadas (solo heterogêneo complexo, ambiente industrial)

MALHA DE ATERRAMENTO:
- Resistência de malha (Sverak, 1979):
  R = ρ/√A + ρ/(Lt) · (1 + 1/(1+h√(20/A)))
  onde: ρ=resistividade (Ω·m), A=área (m²), Lt=comprimento total de condutores (m)
- Tensão de toque e de passo (IEEE 80):
  Etoque = (ρs·Cs·Igs) / (1000+1.5·ρs·Cs)    [limite: 50+0.116/√t]
  Epasso = (ρs·Cs·Igs) / (1000+6·ρs·Cs)       [limite: 50+0.7·(116/√t)]
- GPR (Ground Potential Rise): V_GPR = R_malha × If
- Condutores: cobre nu 50/95/120 mm², cabo cobreado, aço zincado
- Eletrodos: haste Copperweld 5/8" × 2.4m, placa, anel
- Solos típicos: granito seco (>10⁶ Ω·m), areia (10³-10⁴), argila (10-100),
  solo úmido (10-50), solo contaminado (<5)
- Redução de resistividade: bentonita, GEM, solo de carbonato de cálcio

CRITÉRIOS DE QUALIDADE DO AJUSTE:
- Erro RMS < 5%: Excelente
- Erro RMS 5–10%: Bom
- Erro RMS 10–20%: Regular (considerar mais camadas)
- Erro RMS > 20%: Ruim (revisar dados de campo ou geometria da malha)

EQUIPAMENTOS DE MEDIÇÃO:
- Terrômetro Megabras MTR-1520, MTR-1522
- Megger DET4TC2, Fluke 1623-2
- Sensores de 4 terminais (método Wenner/Schlumberger)
- Correção para interferências de estruturas metálicas

Responda sempre em português brasileiro. Seja preciso, técnico e cite as normas
quando relevante. Se o usuário perguntar sobre cálculos específicos da malha,
mostre as fórmulas e o passo a passo. Seja objetivo e profissional."""


def chat_ia_sidebar():
    """Renderiza o painel de Chat IA na sidebar."""
    st.sidebar.divider()
    st.sidebar.subheader("🤖 Assistente IA — Normas de Aterramento")

    with st.sidebar.expander("🔑 Configurar API Key", expanded=not bool(st.session_state.chat_api_key)):
        api_key_input = st.text_input(
            "Anthropic API Key:",
            value=st.session_state.chat_api_key,
            type="password",
            help="Obtenha em console.anthropic.com — começa com 'sk-ant-...'",
            key="api_key_field"
        )
        if api_key_input != st.session_state.chat_api_key:
            st.session_state.chat_api_key = api_key_input
        if st.session_state.chat_api_key:
            st.success("✅ API Key configurada")
        else:
            st.info("Insira sua API Key para ativar o assistente.")

    if not st.session_state.chat_api_key:
        st.sidebar.info("Configure a API Key acima para usar o assistente de IA.")
        return

    # Botão para abrir o chat em tela cheia
    if st.sidebar.button("💬 Abrir Chat Assistente IA", use_container_width=True, type="primary"):
        st.session_state.chat_aberto = True
        st.rerun()

    if st.sidebar.button("🗑️ Limpar Histórico do Chat", use_container_width=True):
        st.session_state.chat_historico = []
        st.rerun()

    n = len(st.session_state.chat_historico)
    if n > 0:
        st.sidebar.caption(f"💬 {n//2} mensagem(ns) no histórico")


def render_chat_modal():
    """Renderiza o chat IA em tela cheia."""
    st.title("🤖 Assistente IA — Engenharia de Aterramento")
    st.caption("Especialista em NBR 7117, NBR 5419, IEEE 80, estratificação de solo e malhas de aterramento")

    col_chat, col_fechar = st.columns([5, 1])
    with col_fechar:
        if st.button("✖ Fechar Chat", use_container_width=True):
            st.session_state.chat_aberto = False
            st.rerun()

    # Histórico de mensagens
    chat_container = st.container(height=500)
    with chat_container:
        if not st.session_state.chat_historico:
            st.markdown("""
**👋 Olá! Sou o assistente de engenharia elétrica da Iluvium.**

Posso ajudar com:
- 📐 **Interpretação de curvas** de estratificação (NBR 7117)
- 🔌 **Cálculo de malha** de aterramento (IEEE 80, NBR 5419)
- ⚡ **Tensões de toque e passo** — critérios de segurança
- 🌱 **Tipos de solo** e resistividades típicas
- 📋 **Normas técnicas** — ABNT, IEEE, IEC
- 🔧 **Dúvidas técnicas** sobre o Motor Iluvium

*Digite sua pergunta abaixo!*
""")
        for msg in st.session_state.chat_historico:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

    # Input
    pergunta = st.chat_input("Digite sua dúvida técnica aqui...")

    if pergunta:
        # Adiciona mensagem do usuário
        st.session_state.chat_historico.append({"role": "user", "content": pergunta})

        # Chama API
        with st.spinner("🤔 Consultando assistente..."):
            try:
                import anthropic
                cliente_ai = anthropic.Anthropic(api_key=st.session_state.chat_api_key)

                # Monta histórico para a API
                msgs_api = [{"role": m["role"], "content": m["content"]}
                            for m in st.session_state.chat_historico]

                resposta = cliente_ai.messages.create(
                    model="claude-sonnet-4-20250514",
                    max_tokens=2048,
                    system=SYSTEM_PROMPT_ENGENHARIA,
                    messages=msgs_api
                )
                conteudo = resposta.content[0].text
            except ImportError:
                conteudo = ("❌ **Biblioteca anthropic não instalada.**\n\n"
                            "Execute no terminal:\n```\npip install anthropic\n```")
            except Exception as e:
                err = str(e)
                if "api_key" in err.lower() or "authentication" in err.lower():
                    conteudo = "❌ **API Key inválida.** Verifique na sidebar."
                elif "rate_limit" in err.lower():
                    conteudo = "⏳ **Limite de requisições atingido.** Aguarde alguns instantes."
                else:
                    conteudo = f"❌ **Erro ao consultar a IA:** {err}"

        st.session_state.chat_historico.append({"role": "assistant", "content": conteudo})
        st.rerun()

# =========================================================================
# CONSTANTES DE UI
# =========================================================================
DIRECOES   = list("ABCDEFGHIJKLMNOP")
# Cores unificadas: COR_EIXO_HEX (acima) é a fonte única de verdade.
# Para letras além de H, recicla a paleta.
def _cor_eixo(letra):
    if letra in COR_EIXO_HEX:
        return COR_EIXO_HEX[letra]
    # Letras I, J, K... reciclam A, B, C...
    base = list(COR_EIXO_HEX.keys())
    idx = (ord(letra.upper()) - ord('A')) % len(base)
    return COR_EIXO_HEX[base[idx]]
COR_MAP    = {d: _cor_eixo(d) for d in DIRECOES}
CORES      = list(COR_MAP.values())
BUSSOLA    = {"🧭 Norte (0°)":0,"↗ NE (45°)":45,"➡ Leste (90°)":90,
              "↘ SE (135°)":135,"⬇ Sul (180°)":180,"↙ SO (225°)":225,
              "⬅ Oeste (270°)":270,"↖ NO (315°)":315}
CORES_SOLO = ['#5D4037','#795548','#A1887F','#D7CCC8']

# =========================================================================
# █████ MÓDULO DE GESTÃO DE LAUDOS v3.3 █████
# =========================================================================
# Sistema de salvar/carregar/buscar laudos.
# Storage: arquivo local JSON (com instruções para o usuário fazer backup
# manual do arquivo .iluvium para o Drive).
#
# Estrutura de cada laudo:
# {
#   "id": "uuid",
#   "metadados": {cliente, projeto, local_obra, data_ensaio, status,
#                 tecnico, crea, criado_em, atualizado_em},
#   "estado": {dados_tabelas, direcoes_ativas, lat, lon, correcao_haste,
#              parecer, ...},
#   "resultado": {camadas, rho_final, erro_rms_pct, erro_rms_log},
#   "logs": [{timestamp, evento, detalhes}]
# }
# =========================================================================
import json, uuid as _uuid, base64
from pathlib import Path

LAUDOS_DIR = Path(tempfile.gettempdir()) / "iluvium_laudos"
LAUDOS_DIR.mkdir(parents=True, exist_ok=True)
LAUDOS_INDEX = LAUDOS_DIR / "_index.json"
RETENCAO_DIAS = 90

def _serializa(obj):
    """Converte DataFrames e tipos não-JSON em dicts."""
    if isinstance(obj, pd.DataFrame):
        return {"__df__": True, "data": obj.to_dict(orient='list'),
                "columns": list(obj.columns)}
    if isinstance(obj, np.ndarray):
        return {"__ndarray__": True, "data": obj.tolist()}
    if isinstance(obj, (np.integer, np.floating)):
        return float(obj)
    if isinstance(obj, datetime):
        return {"__datetime__": True, "iso": obj.isoformat()}
    if isinstance(obj, dict):
        return {k: _serializa(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_serializa(x) for x in obj]
    return obj

def _deserializa(obj):
    """Reverte _serializa."""
    if isinstance(obj, dict):
        if obj.get("__df__"):
            df = pd.DataFrame(obj["data"])
            if "columns" in obj:
                cols_existentes = [c for c in obj["columns"] if c in df.columns]
                if cols_existentes:
                    df = df[cols_existentes]
            return df
        if obj.get("__ndarray__"):
            return np.array(obj["data"])
        if obj.get("__datetime__"):
            return datetime.fromisoformat(obj["iso"])
        return {k: _deserializa(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_deserializa(x) for x in obj]
    return obj

def _carrega_index():
    if not LAUDOS_INDEX.exists():
        return {}
    try:
        return json.loads(LAUDOS_INDEX.read_text(encoding='utf-8'))
    except Exception:
        return {}

def _salva_index(idx):
    LAUDOS_INDEX.write_text(json.dumps(idx, ensure_ascii=False, indent=2),
                            encoding='utf-8')

def _coleta_estado_atual():
    """Captura tudo do session_state que define um laudo."""
    ss = st.session_state
    estado = {
        "dados_tabelas": {k: _serializa(v) for k, v in ss.get("dados_tabelas", {}).items()},
        "direcoes_ativas": ss.get("direcoes_ativas_sel", []),
        "correcao_haste": ss.get("correcao_haste_sel", False),
        "step_plot_ativo": ss.get("step_plot_ativo_sel", True),
        "lat": float(ss.get("lat_atual", -21.2089)),
        "lon": float(ss.get("lon_atual", -50.4328)),
        "map_zoom": ss.get("map_zoom", 15),
        "parecer": ss.get("parecer_atual", ""),
        "desenhos_geojson": ss.get("desenhos_geojson",
                                    {"type":"FeatureCollection","features":[]}),
    }
    resultado = {
        "calc_concluido": ss.get("calc_concluido", False),
        "camadas_atuais": _serializa(ss.get("camadas_atuais", None)),
        "rho_final_calc": _serializa(ss.get("rho_final_calc", None)),
        "erro_rms_pct": float(ss.get("erro_rms_pct", 0)) if ss.get("erro_rms_pct") else None,
        "erro_rms_log": float(ss.get("erro_rms_log", 0)) if ss.get("erro_rms_log") else None,
        "df_r_resultado": _serializa(ss.get("df_r_resultado", None)),
        "df_d_resultado": _serializa(ss.get("df_d_resultado", None)),
    }
    return estado, resultado

def _aplica_estado(estado, resultado):
    """Restaura o session_state a partir de um laudo carregado."""
    ss = st.session_state
    ss.dados_tabelas = {k: _deserializa(v) for k, v in estado.get("dados_tabelas", {}).items()}
    ss.direcoes_ativas_carregadas = estado.get("direcoes_ativas", ["A"])
    ss.correcao_haste_carregado = estado.get("correcao_haste", False)
    ss.step_plot_carregado = estado.get("step_plot_ativo", True)
    ss.gps_lat = estado.get("lat", -21.2089)
    ss.gps_lon = estado.get("lon", -50.4328)
    ss.map_zoom = estado.get("map_zoom", 15)
    ss.parecer_carregado = estado.get("parecer", "")
    ss.desenhos_geojson = estado.get("desenhos_geojson",
                                      {"type":"FeatureCollection","features":[]})
    if resultado.get("calc_concluido"):
        ss.calc_concluido = True
        ss.camadas_atuais = _deserializa(resultado.get("camadas_atuais"))
        ss.rho_final_calc = _deserializa(resultado.get("rho_final_calc"))
        ss.erro_rms_pct = resultado.get("erro_rms_pct")
        ss.erro_rms_log = resultado.get("erro_rms_log")
        ss.df_r_resultado = _deserializa(resultado.get("df_r_resultado"))
        ss.df_d_resultado = _deserializa(resultado.get("df_d_resultado"))

def adicionar_log(laudo_id, evento, detalhes=""):
    """Adiciona uma entrada nos logs de execução de um laudo."""
    arq = LAUDOS_DIR / f"{laudo_id}.json"
    if not arq.exists(): return
    try:
        laudo = json.loads(arq.read_text(encoding='utf-8'))
        laudo.setdefault("logs", []).append({
            "timestamp": datetime.now().isoformat(),
            "evento": evento,
            "detalhes": str(detalhes)[:500]
        })
        # Mantém só os últimos 200 logs por laudo
        laudo["logs"] = laudo["logs"][-200:]
        arq.write_text(json.dumps(laudo, ensure_ascii=False), encoding='utf-8')
    except Exception:
        pass

def salvar_laudo(metadados, laudo_id=None, criar_novo=False):
    """Salva o estado atual como laudo. Se laudo_id já existe, atualiza.
    Retorna o ID do laudo."""
    estado, resultado = _coleta_estado_atual()
    idx = _carrega_index()
    agora = datetime.now().isoformat()

    if laudo_id is None or criar_novo:
        laudo_id = str(_uuid.uuid4())
        criado_em = agora
    else:
        existente = idx.get(laudo_id, {})
        criado_em = existente.get("criado_em", agora)

    laudo = {
        "id": laudo_id,
        "metadados": {**metadados,
                      "criado_em": criado_em,
                      "atualizado_em": agora},
        "estado": estado,
        "resultado": resultado,
    }
    arq = LAUDOS_DIR / f"{laudo_id}.json"

    # Preserva logs se existirem
    if arq.exists() and not criar_novo:
        try:
            antigo = json.loads(arq.read_text(encoding='utf-8'))
            laudo["logs"] = antigo.get("logs", [])
        except Exception:
            laudo["logs"] = []
    else:
        laudo["logs"] = [{
            "timestamp": agora,
            "evento": "criacao",
            "detalhes": f"Laudo criado por {metadados.get('tecnico','?')}"
        }]

    arq.write_text(json.dumps(laudo, ensure_ascii=False), encoding='utf-8')

    idx[laudo_id] = {
        **{k: metadados.get(k, "") for k in
           ["cliente", "projeto", "local_obra", "data_ensaio",
            "status", "tecnico", "crea"]},
        "criado_em": criado_em,
        "atualizado_em": agora,
        "rms_pct": resultado.get("erro_rms_pct"),
    }
    _salva_index(idx)
    return laudo_id

def carregar_laudo(laudo_id):
    """Carrega um laudo do disco e aplica ao session_state."""
    arq = LAUDOS_DIR / f"{laudo_id}.json"
    if not arq.exists():
        return False, "Arquivo do laudo não encontrado."
    try:
        laudo = json.loads(arq.read_text(encoding='utf-8'))
        _aplica_estado(laudo.get("estado", {}), laudo.get("resultado", {}))
        st.session_state.laudo_atual_id = laudo_id
        st.session_state.laudo_atual_metadados = laudo.get("metadados", {})
        adicionar_log(laudo_id, "carregamento",
                      f"Laudo carregado para edicao")
        return True, "OK"
    except Exception as e:
        return False, str(e)

def excluir_laudo(laudo_id):
    """Remove o laudo permanentemente."""
    arq = LAUDOS_DIR / f"{laudo_id}.json"
    if arq.exists():
        arq.unlink()
    idx = _carrega_index()
    idx.pop(laudo_id, None)
    _salva_index(idx)

def duplicar_laudo(laudo_id):
    """Cria uma cópia do laudo com novo ID."""
    arq = LAUDOS_DIR / f"{laudo_id}.json"
    if not arq.exists(): return None
    try:
        original = json.loads(arq.read_text(encoding='utf-8'))
        novo_id = str(_uuid.uuid4())
        agora = datetime.now().isoformat()
        original["id"] = novo_id
        original["metadados"]["projeto"] = original["metadados"].get("projeto","") + " (cópia)"
        original["metadados"]["status"] = "Rascunho"
        original["metadados"]["criado_em"] = agora
        original["metadados"]["atualizado_em"] = agora
        original["logs"] = [{
            "timestamp": agora, "evento": "duplicacao",
            "detalhes": f"Duplicado de {laudo_id}"
        }]
        (LAUDOS_DIR / f"{novo_id}.json").write_text(
            json.dumps(original, ensure_ascii=False), encoding='utf-8')
        idx = _carrega_index()
        idx[novo_id] = {
            **{k: original["metadados"].get(k,"") for k in
               ["cliente","projeto","local_obra","data_ensaio",
                "status","tecnico","crea"]},
            "criado_em": agora, "atualizado_em": agora,
            "rms_pct": original.get("resultado",{}).get("erro_rms_pct"),
        }
        _salva_index(idx)
        return novo_id
    except Exception:
        return None

def listar_laudos():
    """Retorna lista de todos os laudos com metadados (para a tela de listagem)."""
    idx = _carrega_index()
    laudos = []
    for lid, meta in idx.items():
        laudos.append({"id": lid, **meta})
    laudos.sort(key=lambda x: x.get("atualizado_em",""), reverse=True)
    return laudos

def aplica_retencao_90d():
    """Identifica laudos com mais de 90 dias e retorna lista para alertar.
    NÃO apaga automaticamente — só sinaliza."""
    idx = _carrega_index()
    agora = datetime.now()
    a_apagar = []
    for lid, meta in idx.items():
        try:
            atualizado = datetime.fromisoformat(meta.get("atualizado_em",""))
            dias = (agora - atualizado).days
            if dias >= RETENCAO_DIAS:
                a_apagar.append({"id": lid, "dias": dias, **meta})
        except Exception:
            continue
    return a_apagar

def exportar_iluvium():
    """Cria arquivo .iluvium (ZIP) com TODOS os laudos para backup manual."""
    import zipfile, io
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, 'w', zipfile.ZIP_DEFLATED) as zf:
        if LAUDOS_INDEX.exists():
            zf.write(LAUDOS_INDEX, "_index.json")
        for arq in LAUDOS_DIR.glob("*.json"):
            if arq.name != "_index.json":
                zf.write(arq, arq.name)
        manifest = {
            "tipo": "iluvium_backup",
            "versao": "3.3",
            "exportado_em": datetime.now().isoformat(),
            "total_laudos": len(list(LAUDOS_DIR.glob("*.json"))) - 1
        }
        zf.writestr("_manifest.json", json.dumps(manifest, indent=2))
    return buf.getvalue()

def importar_iluvium(zip_bytes, modo="merge"):
    """Importa um .iluvium. modo='merge' adiciona, modo='replace' substitui tudo."""
    import zipfile, io
    try:
        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
            nomes = zf.namelist()
            if "_manifest.json" not in nomes:
                return False, "Arquivo .iluvium inválido (sem manifest)."

            if modo == "replace":
                for arq in LAUDOS_DIR.glob("*.json"):
                    arq.unlink()

            for nome in nomes:
                if nome.startswith("_") and nome != "_index.json":
                    continue
                conteudo = zf.read(nome).decode('utf-8')
                if nome == "_index.json":
                    novo_idx = json.loads(conteudo)
                    if modo == "replace":
                        _salva_index(novo_idx)
                    else:
                        atual = _carrega_index()
                        atual.update(novo_idx)
                        _salva_index(atual)
                else:
                    (LAUDOS_DIR / nome).write_text(conteudo, encoding='utf-8')
            return True, f"Importados {len(nomes)-2} laudos."
    except Exception as e:
        return False, f"Erro ao importar: {e}"

def filtrar_laudos(laudos, termo_busca="", status_filtro="Todos",
                   data_de=None, data_ate=None):
    resultado = []
    termo = termo_busca.lower().strip()
    for laudo in laudos:
        if termo:
            campos = " ".join([str(laudo.get(k,"")).lower() for k in
                              ["cliente","projeto","local_obra","tecnico"]])
            if termo not in campos:
                continue
        if status_filtro != "Todos" and laudo.get("status") != status_filtro:
            continue
        if data_de or data_ate:
            try:
                d = datetime.fromisoformat(laudo.get("data_ensaio","")[:10])
                if data_de and d < datetime.combine(data_de, datetime.min.time()):
                    continue
                if data_ate and d > datetime.combine(data_ate, datetime.max.time()):
                    continue
            except Exception:
                pass
        resultado.append(laudo)
    return resultado

# ═════════════════════ AUTO-SAVE v3.8 ═════════════════════
# Dois níveis de persistência:
#   1. Auto-save do LAUDO ABERTO no índice de laudos (a cada 5 min)
#   2. Arquivo de RECUPERAÇÃO DE EMERGÊNCIA (salvo sempre, mesmo sem laudo)
#      → sobrevive ao fechamento do app; oferece restauração ao reabrir

# Caminho do arquivo de emergência (ao lado do main.py)
try:
    _MAIN_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
except Exception:
    _MAIN_DIR = Path.cwd()

# Arquivo de emergência em AppData (persiste entre sessões, nunca é limpo pelo SO)
# Windows: C:\Users\<user>\AppData\Local\Iluvium\
# Linux/Mac (Streamlit Cloud): ~/.local/share/Iluvium/
try:
    _APPDATA = Path(os.environ.get("LOCALAPPDATA") or
                    Path.home() / ".local" / "share") / "Iluvium"
    _APPDATA.mkdir(parents=True, exist_ok=True)
except Exception:
    _APPDATA = _MAIN_DIR
EMERGENCIA_PATH = _APPDATA / "iluvium_recovery.json"
EMERGENCIA_INTERVAL = 300   # 5 minutos em segundos

def salvar_emergencia():
    """Salva estado completo em arquivo de recuperação persistente.
    Funciona SEMPRE, mesmo sem laudo aberto — sobrevive ao fechamento do app."""
    try:
        ss = st.session_state
        snapshot = {
            "ts": datetime.now().isoformat(),
            "versao": "3.8",
            "laudo_id": ss.get("laudo_atual_id"),
            "metadados": ss.get("laudo_atual_metadados", {}),
            "dados_tabelas": {k: _serializa(v)
                              for k, v in ss.get("dados_tabelas", {}).items()},
            "direcoes_ativas": ss.get("direcoes_ativas_sel", ["A"]),
            "lat": float(ss.get("lat_atual", -21.2089)),
            "lon": float(ss.get("lon_atual", -50.4328)),
            "parecer": ss.get("parecer_atual", ""),
            "desenhos_geojson": ss.get("desenhos_geojson",
                                       {"type": "FeatureCollection", "features": []}),
            "calc_concluido": ss.get("calc_concluido", False),
            "camadas_atuais": _serializa(ss.get("camadas_atuais")),
            "erro_rms_pct": ss.get("erro_rms_pct"),
            "df_r_resultado": _serializa(ss.get("df_r_resultado")),
            "df_d_resultado": _serializa(ss.get("df_d_resultado")),
        }
        EMERGENCIA_PATH.write_text(
            json.dumps(snapshot, ensure_ascii=False),
            encoding='utf-8')
        return True
    except Exception:
        return False

def carregar_emergencia():
    """Tenta carregar o arquivo de recuperação de emergência.
    Retorna (dict, str_data) ou (None, None)."""
    if not EMERGENCIA_PATH.exists():
        return None, None
    try:
        snap = json.loads(EMERGENCIA_PATH.read_text(encoding='utf-8'))
        ts_str = snap.get("ts", "")
        try:
            ts_fmt = datetime.fromisoformat(ts_str).strftime("%d/%m/%Y %H:%M")
        except Exception:
            ts_fmt = ts_str
        return snap, ts_fmt
    except Exception:
        return None, None

def restaurar_emergencia(snap):
    """Restaura o session_state a partir de um snapshot de emergência."""
    ss = st.session_state
    ss.dados_tabelas = {k: _deserializa(v)
                        for k, v in snap.get("dados_tabelas", {}).items()}
    if not ss.dados_tabelas:
        ss.dados_tabelas = {"A": df_novo_eixo()}
    ss.direcoes_ativas_carregadas = snap.get("direcoes_ativas", ["A"])
    ss.gps_lat  = snap.get("lat", -21.2089)
    ss.gps_lon  = snap.get("lon", -50.4328)
    ss.parecer_carregado   = snap.get("parecer", "")
    ss.desenhos_geojson    = snap.get("desenhos_geojson",
                                       {"type": "FeatureCollection", "features": []})
    ss.laudo_atual_id      = snap.get("laudo_id")
    ss.laudo_atual_metadados = snap.get("metadados", {})
    if snap.get("calc_concluido"):
        ss.calc_concluido  = True
        ss.camadas_atuais  = _deserializa(snap.get("camadas_atuais"))
        ss.erro_rms_pct    = snap.get("erro_rms_pct")
        ss.df_r_resultado  = _deserializa(snap.get("df_r_resultado"))
        ss.df_d_resultado  = _deserializa(snap.get("df_d_resultado"))

def excluir_emergencia():
    """Remove o arquivo de recuperação (após restauração bem-sucedida ou descarte)."""
    try:
        if EMERGENCIA_PATH.exists():
            EMERGENCIA_PATH.unlink()
    except Exception:
        pass

def autosave_se_necessario(intervalo_seg=EMERGENCIA_INTERVAL):
    """Auto-save em dois níveis:
    1. Salva o laudo aberto no índice (se houver)
    2. Salva arquivo de recuperação de emergência (sempre)
    Intervalo: 5 minutos."""
    if not st.session_state.get("autosave_ativo", True):
        return
    agora = time.time()
    ultimo = st.session_state.get("ultimo_autosave", 0)
    if agora - ultimo < intervalo_seg:
        return

    # Nível 1: laudo aberto
    laudo_id = st.session_state.get("laudo_atual_id")
    if laudo_id:
        metadados = st.session_state.get("laudo_atual_metadados", {})
        if metadados.get("cliente"):
            try:
                salvar_laudo(metadados, laudo_id=laudo_id)
                adicionar_log(laudo_id, "autosave", "Auto-save (5 min)")
            except Exception:
                pass

    # Nível 2: arquivo de emergência (sempre)
    salvar_emergencia()

    st.session_state.ultimo_autosave = agora
    st.session_state.autosave_ts = datetime.now().strftime("%H:%M:%S")

def salvar_manual():
    """Salvo pelo botão explícito — salva laudo + emergência + loga."""
    laudo_id = st.session_state.get("laudo_atual_id")
    if laudo_id:
        metadados = st.session_state.get("laudo_atual_metadados", {})
        salvar_laudo(metadados, laudo_id=laudo_id)
        adicionar_log(laudo_id, "save_manual", "Salvo manualmente pelo botão")
    salvar_emergencia()
    st.session_state.autosave_ts = datetime.now().strftime("%H:%M:%S")
    st.session_state.ultimo_autosave = time.time()
    return True

# =========================================================================
# CONFIGURAÇÃO DA PÁGINA
# =========================================================================
st.set_page_config(page_title="Iluvium | Motor Master v3.8",
                   layout="wide", initial_sidebar_state="expanded")

# ── Se chat aberto, renderiza apenas o chat ──────────────────────────────
if 'chat_aberto' not in st.session_state:
    st.session_state.chat_aberto = False

if st.session_state.chat_aberto:
    render_chat_modal()
    st.stop()

# =========================================================================
# SIDEBAR
# =========================================================================
with st.sidebar:
    st.title("⚡ Iluvium v3.8")

    # ── NAVEGAÇÃO PRINCIPAL ──────────────────────────────────────────────
    # Sistema de redirect: outras partes do app setam _redirect_to,
    # que é processado AQUI antes do widget ser renderizado.
    # (Não é possível modificar st.session_state.pagina_atual após o
    # widget radio com essa mesma key ser instanciado.)
    if "_redirect_to" in st.session_state:
        st.session_state.pagina_atual = st.session_state.pop("_redirect_to")

    pagina = st.radio("📂 Navegação", ["🔬 Calculadora", "📁 Meus Laudos"],
                      label_visibility="collapsed",
                      key="pagina_atual")
    st.divider()

    # ── INDICADOR DE LAUDO ATUAL ─────────────────────────────────────────
    if st.session_state.get("laudo_atual_id"):
        meta = st.session_state.get("laudo_atual_metadados", {})
        cliente = meta.get("cliente", "—")[:18]
        projeto = meta.get("projeto", "—")[:22]
        ts = st.session_state.get("autosave_ts")
        st.success(f"📝 **Editando:** {cliente}\n\n*{projeto}*"
                   + (f"\n\n💾 Auto-save: {ts}" if ts else ""))
        if st.button("🚪 Fechar laudo", use_container_width=True,
                     help="Volta ao modo livre (sem laudo aberto)"):
            for k in ["laudo_atual_id","laudo_atual_metadados",
                      "ultimo_autosave","autosave_ts","desenhos_geojson"]:
                st.session_state.pop(k, None)
            st.rerun()
        st.divider()

    st.title("⚙️ Configurações Técnicas")
    st.divider()
    st.subheader("🔧 Física da Medição")
    correcao_haste  = st.checkbox("Correção de Haste (Palmer)", value=False,
                                  on_change=reseta_calculo)
    step_plot_ativo = st.checkbox("Step Plot (Perfil em Degraus)", value=True)
    st.divider()
    st.subheader("🌍 Coordenadas GPS")

    # Inicializa coordenadas padrão se não existirem
    if 'gps_lat' not in st.session_state:
        st.session_state.gps_lat = -21.2089
    if 'gps_lon' not in st.session_state:
        st.session_state.gps_lon = -50.4328

    with st.expander("📍 Capturar Localização", expanded=False):
        try:
            from streamlit_geolocation import streamlit_geolocation
            g = streamlit_geolocation()
            if (isinstance(g, dict)
                    and g.get('latitude') is not None
                    and g.get('longitude') is not None):
                _lat_g = float(g['latitude'])
                _lon_g = float(g['longitude'])
                st.success(f"📡 {_lat_g:.6f}, {_lon_g:.6f}")
                if st.button("✅ Usar esta localização", type="primary",
                             use_container_width=True, key="btn_usar_gps"):
                    # Seta as keys dos widgets → atualiza campos na próxima renderização
                    st.session_state['coord_lat'] = _lat_g
                    st.session_state['coord_lon'] = _lon_g
                    st.session_state.lat_atual = _lat_g
                    st.session_state.lon_atual = _lon_g
                    st.session_state.gps_lat   = _lat_g
                    st.session_state.gps_lon   = _lon_g
                    st.rerun()
        except Exception:
            st.info("Componente streamlit-geolocation não disponível. "
                    "Digite as coordenadas manualmente abaixo.")

    # Solução definitiva: key= fixo + setar session_state ANTES do widget
    # Quando GPS confirma, seta st.session_state['coord_lat'] = novo_valor
    # O widget com key='coord_lat' lê esse valor no próximo ciclo
    if 'coord_lat' not in st.session_state:
        st.session_state['coord_lat'] = -21.2089
    if 'coord_lon' not in st.session_state:
        st.session_state['coord_lon'] = -50.4328

    lat_input = st.number_input(
        "Latitude", format="%.6f", step=0.000001,
        key="coord_lat")
    lon_input = st.number_input(
        "Longitude", format="%.6f", step=0.000001,
        key="coord_lon")

    # Persiste para uso em todo o app
    st.session_state.lat_atual = lat_input
    st.session_state.lon_atual = lon_input
    st.session_state.gps_lat   = lat_input
    st.session_state.gps_lon   = lon_input
    st.divider()
    with st.expander("ℹ️ Motor v3.8"):
        st.markdown("""
**v3.3 — Sistema de Laudos:**
- 📁 Meus Laudos (CRUD completo) ✅
- 💾 Auto-save (30s) ✅
- 📤 Backup .iluvium ✅
- 🗑 Retenção 90 dias ✅

**v3.2 — Pseudo-Seção:**
- Paleta Res2DInv ✅
- Máscara cone-de-influência ✅
- RMS no título ✅

**v3.1 — Correções:**
- KeyError unicode ✅
- Escopo df_d ✅
- Chat IA integrado ✅
""")

# ── Chat IA na sidebar ───────────────────────────────────────────────────
chat_ia_sidebar()

# =========================================================================
# INICIALIZA TABELAS
# =========================================================================
if "A" not in st.session_state.dados_tabelas:
    st.session_state.dados_tabelas["A"] = df_novo_eixo()

# =========================================================================
# █████ PÁGINA: MEUS LAUDOS █████
# =========================================================================
if st.session_state.get("pagina_atual") == "📁 Meus Laudos":
    st.title("📁 Meus Laudos")
    st.caption(f"Laudos salvos no dispositivo · Retenção: {RETENCAO_DIAS} dias · Backup recomendado periodicamente")

    # ── Alerta de retenção (90 dias) ─────────────────────────────────────
    a_apagar = aplica_retencao_90d()
    if a_apagar:
        with st.expander(f"⚠️ {len(a_apagar)} laudo(s) com mais de {RETENCAO_DIAS} dias", expanded=True):
            st.warning("Estes laudos passaram do limite de retenção. "
                       "Faça backup (.iluvium) ou eles podem ser excluídos.")
            for la in a_apagar:
                st.markdown(f"- **{la.get('cliente','—')}** · {la.get('projeto','—')} · "
                            f"{la.get('dias',0)} dias atrás")

    # ── Botões de ação topo ──────────────────────────────────────────────
    cb1, cb2, cb3, cb4 = st.columns(4)
    with cb1:
        if st.button("➕ Novo Laudo", type="primary", use_container_width=True):
            st.session_state.criar_laudo_modal = True
    with cb2:
        bk = exportar_iluvium()
        st.download_button("📤 Exportar Backup",
            data=bk,
            file_name=f"iluvium_backup_{datetime.now().strftime('%Y%m%d_%H%M')}.iluvium",
            mime="application/zip",
            use_container_width=True,
            help="Baixa um arquivo .iluvium com TODOS os laudos. Salve no Google Drive.")
    with cb3:
        upload_bk = st.file_uploader("📥 Importar .iluvium",
                                      type=["iluvium","zip"],
                                      label_visibility="collapsed",
                                      key="upload_iluvium")
        if upload_bk is not None:
            modo_imp = st.radio("Modo:", ["Adicionar (merge)","Substituir tudo"],
                                horizontal=True, key="modo_imp")
            if st.button("✅ Confirmar importação", use_container_width=True):
                ok, msg = importar_iluvium(upload_bk.read(),
                    modo="merge" if "merge" in modo_imp else "replace")
                if ok:
                    st.success(msg); time.sleep(1); st.rerun()
                else:
                    st.error(msg)
    with cb4:
        st.metric("Total de laudos", len(listar_laudos()))

    st.divider()

    # ── Modal de criação de novo laudo ───────────────────────────────────
    if st.session_state.get("criar_laudo_modal"):
        with st.container(border=True):
            st.subheader("➕ Criar Novo Laudo")
            cc1, cc2 = st.columns(2)
            cliente = cc1.text_input("Cliente / Empresa *", key="novo_cliente")
            projeto = cc2.text_input("Projeto *", key="novo_projeto",
                                      value="Malha de Aterramento")
            cc3, cc4 = st.columns(2)
            local_obra = cc3.text_input("Local da Obra *", key="novo_local",
                                         placeholder="Ex: SE Aracatuba 138/13.8 kV")
            data_ensaio = cc4.date_input("Data do Ensaio *",
                                          value=datetime.now().date(),
                                          key="novo_data")
            cc5, cc6 = st.columns(2)
            tecnico = cc5.text_input("Engenheiro Responsável", key="novo_tecnico")
            crea = cc6.text_input("CREA/CFT", key="novo_crea")
            status = st.selectbox("Status inicial",
                ["Rascunho","Em Revisão","Finalizado"], key="novo_status")

            cb_ok, cb_cancel = st.columns([1,1])
            if cb_ok.button("✅ Criar e abrir laudo", type="primary",
                            use_container_width=True):
                if not (cliente and projeto and local_obra):
                    st.error("Preencha os campos obrigatórios (*)")
                else:
                    metadados = {
                        "cliente": cliente, "projeto": projeto,
                        "local_obra": local_obra,
                        "data_ensaio": data_ensaio.isoformat(),
                        "status": status,
                        "tecnico": tecnico, "crea": crea
                    }
                    # Reset do estado antes de criar (laudo limpo)
                    reseta_calculo()
                    st.session_state.dados_tabelas = {"A": df_novo_eixo()}
                    laudo_id = salvar_laudo(metadados, criar_novo=True)
                    st.session_state.laudo_atual_id = laudo_id
                    st.session_state.laudo_atual_metadados = metadados
                    st.session_state.pop("criar_laudo_modal", None)
                    st.session_state._redirect_to = "🔬 Calculadora"
                    st.success(f"✅ Laudo criado! Redirecionando...")
                    time.sleep(0.8); st.rerun()
            if cb_cancel.button("❌ Cancelar", use_container_width=True):
                st.session_state.pop("criar_laudo_modal", None)
                st.rerun()
        st.stop()

    # ── Filtros e busca ──────────────────────────────────────────────────
    cf1, cf2, cf3 = st.columns([2, 1, 1])
    with cf1:
        busca = st.text_input("🔍 Buscar (cliente, projeto, local)",
                              key="busca_laudos", placeholder="Digite para filtrar...")
    with cf2:
        status_filtro = st.selectbox("Status",
            ["Todos","Rascunho","Em Revisão","Finalizado"], key="filtro_status")
    with cf3:
        ordem = st.selectbox("Ordenar por",
            ["Mais recentes","Mais antigos","Cliente A→Z","RMS ↑"],
            key="filtro_ordem")

    # ── Lista de laudos ──────────────────────────────────────────────────
    laudos = listar_laudos()
    laudos_filtrados = filtrar_laudos(laudos, busca, status_filtro)

    if ordem == "Mais antigos":
        laudos_filtrados.sort(key=lambda x: x.get("atualizado_em",""))
    elif ordem == "Cliente A→Z":
        laudos_filtrados.sort(key=lambda x: x.get("cliente","").lower())
    elif ordem == "RMS ↑":
        laudos_filtrados.sort(key=lambda x: (x.get("rms_pct") or 999))

    if not laudos_filtrados:
        if not laudos:
            st.info("📭 Nenhum laudo salvo ainda. Clique em **➕ Novo Laudo** para começar.")
        else:
            st.info("Nenhum laudo corresponde aos filtros aplicados.")
    else:
        st.caption(f"Exibindo **{len(laudos_filtrados)}** de {len(laudos)} laudos")
        for la in laudos_filtrados:
            with st.container(border=True):
                col_info, col_acoes = st.columns([3, 2])
                with col_info:
                    st.markdown(f"### {la.get('cliente','—')} · *{la.get('projeto','—')}*")
                    sub = []
                    if la.get('local_obra'):
                        sub.append(f"📍 {la['local_obra']}")
                    if la.get('data_ensaio'):
                        try:
                            d = datetime.fromisoformat(la['data_ensaio'][:10])
                            sub.append(f"📅 {d.strftime('%d/%m/%Y')}")
                        except: pass
                    if la.get('tecnico'):
                        sub.append(f"👷 {la['tecnico']}")
                    st.caption(" · ".join(sub) if sub else "")

                    bd1, bd2, bd3 = st.columns(3)
                    status_val = la.get("status","—")
                    cor_status = {"Rascunho":"🟡","Em Revisão":"🔵","Finalizado":"🟢"}.get(status_val,"⚪")
                    bd1.markdown(f"**Status:** {cor_status} {status_val}")
                    rms_v = la.get("rms_pct")
                    bd2.markdown(f"**RMS:** {rms_v:.2f}%" if rms_v else "**RMS:** —")
                    try:
                        atu = datetime.fromisoformat(la.get('atualizado_em',''))
                        bd3.markdown(f"**Atualizado:** {atu.strftime('%d/%m %H:%M')}")
                    except:
                        bd3.markdown("**Atualizado:** —")

                with col_acoes:
                    ba1, ba2 = st.columns(2)
                    if ba1.button("📂 Carregar", key=f"carregar_{la['id']}",
                                   use_container_width=True, type="primary"):
                        ok, msg = carregar_laudo(la['id'])
                        if ok:
                            st.session_state._redirect_to = "🔬 Calculadora"
                            st.rerun()
                        else:
                            st.error(msg)
                    if ba2.button("📋 Duplicar", key=f"dup_{la['id']}",
                                   use_container_width=True):
                        novo = duplicar_laudo(la['id'])
                        if novo:
                            st.success("Duplicado!"); time.sleep(0.5); st.rerun()
                    ba3, ba4 = st.columns(2)
                    if ba3.button("📜 Logs", key=f"logs_{la['id']}",
                                   use_container_width=True):
                        st.session_state[f"ver_logs_{la['id']}"] = \
                            not st.session_state.get(f"ver_logs_{la['id']}", False)
                    if ba4.button("🗑️ Excluir", key=f"del_{la['id']}",
                                   use_container_width=True):
                        st.session_state[f"confirmar_del_{la['id']}"] = True

                # Logs expandidos
                if st.session_state.get(f"ver_logs_{la['id']}"):
                    arq = LAUDOS_DIR / f"{la['id']}.json"
                    if arq.exists():
                        try:
                            full = json.loads(arq.read_text(encoding='utf-8'))
                            logs = full.get("logs", [])
                            if logs:
                                st.markdown("**📜 Logs de execução:**")
                                df_logs = pd.DataFrame(logs[-20:])
                                df_logs["timestamp"] = pd.to_datetime(df_logs["timestamp"]).dt.strftime("%d/%m %H:%M:%S")
                                st.dataframe(df_logs, use_container_width=True, hide_index=True)
                            else:
                                st.caption("Nenhum log registrado ainda.")
                        except Exception as e:
                            st.error(f"Erro ao ler logs: {e}")

                # Confirmação de exclusão
                if st.session_state.get(f"confirmar_del_{la['id']}"):
                    st.warning("⚠️ **Tem certeza?** Esta ação é irreversível.")
                    cd1, cd2 = st.columns(2)
                    if cd1.button("✅ Sim, excluir", key=f"sim_del_{la['id']}",
                                   use_container_width=True):
                        excluir_laudo(la['id'])
                        st.session_state.pop(f"confirmar_del_{la['id']}", None)
                        st.success("Laudo excluído."); time.sleep(0.5); st.rerun()
                    if cd2.button("❌ Cancelar", key=f"nao_del_{la['id']}",
                                   use_container_width=True):
                        st.session_state.pop(f"confirmar_del_{la['id']}", None)
                        st.rerun()

    st.stop()  # Não renderiza o restante (calculadora) quando está em Meus Laudos

# =========================================================================
# CABEÇALHO
# =========================================================================
st.title("⚡ Iluvium Engenharia | Motor Master v3.8")
st.caption("Motor geofísico v3.0 + Pseudo-Seção 2D + Sistema de Laudos")

# ── Barra de status do laudo aberto + botões de salvar ───────────────────
laudo_aberto_id = st.session_state.get("laudo_atual_id")
if laudo_aberto_id:
    meta = st.session_state.get("laudo_atual_metadados", {})
    with st.container(border=True):
        cs1, cs2, cs3, cs4 = st.columns([3, 1, 1, 1])
        with cs1:
            st.markdown(f"📝 **Editando laudo:** {meta.get('cliente','—')} · "
                        f"*{meta.get('projeto','—')}* · "
                        f"📍 {meta.get('local_obra','—')}")
            ts_save = st.session_state.get("autosave_ts")
            if ts_save:
                st.caption(f"💾 Auto-save em {ts_save}")
        with cs2:
            if st.button("💾 Salvar agora", use_container_width=True,
                         help="Salva laudo + arquivo de recuperação (Ctrl+S não disponível no browser)"):
                try:
                    salvar_manual()
                    st.toast("✅ Laudo salvo!", icon="💾")
                except Exception as e:
                    st.error(f"Erro: {e}")
        with cs3:
            with st.popover("✏️ Editar metadados", use_container_width=True):
                novo_cli = st.text_input("Cliente", value=meta.get("cliente",""),
                                          key="ed_cli")
                novo_pro = st.text_input("Projeto", value=meta.get("projeto",""),
                                          key="ed_pro")
                novo_loc = st.text_input("Local da Obra",
                                          value=meta.get("local_obra",""),
                                          key="ed_loc")
                novo_st  = st.selectbox("Status",
                    ["Rascunho","Em Revisão","Finalizado"],
                    index=["Rascunho","Em Revisão","Finalizado"].index(
                        meta.get("status","Rascunho")) if meta.get("status") in
                        ["Rascunho","Em Revisão","Finalizado"] else 0,
                    key="ed_st")
                if st.button("Atualizar", type="primary",
                             use_container_width=True, key="btn_atu_meta"):
                    novos_meta = {**meta, "cliente": novo_cli,
                                  "projeto": novo_pro, "local_obra": novo_loc,
                                  "status": novo_st}
                    salvar_laudo(novos_meta, laudo_id=laudo_aberto_id)
                    st.session_state.laudo_atual_metadados = novos_meta
                    st.toast("Metadados atualizados", icon="✏️"); st.rerun()
        with cs4:
            if st.button("🚪 Fechar", use_container_width=True,
                         help="Volta ao modo livre"):
                for k in ["laudo_atual_id","laudo_atual_metadados",
                          "ultimo_autosave","autosave_ts","desenhos_geojson"]:
                    st.session_state.pop(k, None)
                st.rerun()
else:
    # Aviso suave: sem laudo aberto
    cw1, cw2, cw3 = st.columns([3, 1, 1])
    cw1.info("💡 **Modo livre** — você está calculando sem um laudo aberto. "
             "Para salvar este trabalho, crie um laudo em **📁 Meus Laudos**.")
    if cw2.button("➕ Criar laudo agora", use_container_width=True):
        st.session_state._redirect_to = "📁 Meus Laudos"
        st.session_state.criar_laudo_modal = True
        st.rerun()
    if cw3.button("💾 Salvar rascunho", use_container_width=True,
                  help=f"Salva em: {EMERGENCIA_PATH}"):
        if salvar_emergencia():
            st.toast(f"Salvo em AppData\\Iluvium\\", icon="💾")
        else:
            st.toast("Erro ao salvar", icon="⚠️")

# ── Oferta de recuperação (se houver arquivo de emergência) ──────────────
if not st.session_state.get("recovery_checado"):
    st.session_state.recovery_checado = True
    _snap, _ts = carregar_emergencia()
    if _snap and not st.session_state.get("laudo_atual_id"):
        st.session_state.recovery_pendente = (_snap, _ts)

if st.session_state.get("recovery_pendente"):
    _snap, _ts = st.session_state.recovery_pendente
    _meta = _snap.get("metadados", {})
    _cli  = _meta.get("cliente", "—")
    _proj = _meta.get("projeto", "—")
    with st.container(border=True):
        st.warning(
            f"⚠️ **Recuperação disponível** — Auto-save de **{_ts}**\n\n"
            f"Cliente: **{_cli}** · Projeto: **{_proj}**\n\n"
            "O app foi fechado antes de salvar completamente. Deseja restaurar?")
        _rc1, _rc2, _rc3 = st.columns(3)
        if _rc1.button("✅ Restaurar sessão", type="primary",
                       use_container_width=True):
            restaurar_emergencia(_snap)
            excluir_emergencia()
            st.session_state.pop("recovery_pendente")
            st.toast("Sessão restaurada!", icon="✅")
            st.rerun()
        if _rc2.button("❌ Descartar", use_container_width=True):
            excluir_emergencia()
            st.session_state.pop("recovery_pendente")
            st.rerun()
        _rc3.caption(f"📂 `{EMERGENCIA_PATH}`\n\n{_ts}")

# ── Auto-save (a cada 5 min) + arquivo de emergência ─────────────────────
autosave_se_necessario()   # intervalo padrão = 5 min (EMERGENCIA_INTERVAL)

direcoes_ativas = st.multiselect("📐 Direções Ativas (Trenas):", DIRECOES,
                                  default=["A"], on_change=reseta_calculo)
if not direcoes_ativas:
    st.warning("Selecione pelo menos uma direção."); st.stop()

# =========================================================================
# ABAS DE DADOS DE CAMPO
# =========================================================================
# CSS dinâmico para colorir as abas de cada eixo
_css_abas = "<style>\n"
for _i, _d in enumerate(direcoes_ativas):
    _cor_d = COR_MAP.get(_d, "#666")
    _css_abas += (
        f"div[data-baseweb=\"tab-list\"] button[data-baseweb=\"tab\"]:nth-child({_i+1}) {{\n"
        f"  border-bottom: 4px solid {_cor_d}50 !important;\n"
        f"}}\n"
        f"div[data-baseweb=\"tab-list\"] button[data-baseweb=\"tab\"]:nth-child({_i+1})[aria-selected=\"true\"] {{\n"
        f"  border-bottom: 4px solid {_cor_d} !important;\n"
        f"  color: {_cor_d} !important;\n"
        f"  font-weight: 700 !important;\n"
        f"}}\n"
    )
_css_abas += "</style>"
st.markdown(_css_abas, unsafe_allow_html=True)

abas = st.tabs([f"📏 Eixo {d}" for d in direcoes_ativas])
for i, d in enumerate(direcoes_ativas):
    if d not in st.session_state.dados_tabelas:
        st.session_state.dados_tabelas[d] = df_novo_eixo()
    with abas[i]:
        # Banner colorido com a identidade visual do eixo
        _cor_eixo = COR_MAP.get(d, "#666")
        st.markdown(
            f"<div style='background:linear-gradient(90deg,{_cor_eixo} 0%,{_cor_eixo}aa 100%);"
            f"color:white;padding:8px 16px;border-radius:6px;margin-bottom:10px;"
            f"font-weight:600;display:flex;justify-content:space-between;align-items:center;"
            f"font-family:sans-serif;'>"
            f"<span style='font-size:18px'>📏 Eixo {d}</span>"
            f"<span style='font-size:13px;opacity:0.9'>Cor identificadora</span>"
            f"</div>",
            unsafe_allow_html=True
        )
        col_tab, col_mod = st.columns([1.8, 1])
        with col_mod:
            try:
                st.image("MODELO WENNER.png", use_container_width=True)
            except Exception:
                st.info("📷 Coloque 'MODELO WENNER.png' na pasta do projeto")
            st.markdown("**Wenner:** A–M–N–B (espaç. = a)  \nρ_a = 2πa·R | Prof. ≈ 0.5–1.0×a")
        with col_tab:
            # Exibe com labels bonitos, salva com nomes ASCII-safe
            df_interno = st.session_state.dados_tabelas[d].copy()
            df_exibir  = df_para_exibicao(df_interno)
            df_ed_exib = st.data_editor(df_exibir, num_rows="dynamic",
                key=f"tab_{d}", use_container_width=True,
                column_config={
                    LBL_A:   st.column_config.NumberColumn(min_value=0.01, step=0.5),
                    LBL_P:   st.column_config.NumberColumn(min_value=0.0,  step=0.05),
                    LBL_R:   st.column_config.NumberColumn(min_value=0.0),
                    LBL_RHO: st.column_config.NumberColumn(min_value=0.0),
                })
            df_ed = df_de_exibicao(df_ed_exib)

            if not df_ed.equals(df_interno):
                mod = False
                for idx in df_ed.index:
                    a_n   = parse_br_float(df_ed.at[idx, COL_A])
                    p_n   = parse_br_float(df_ed.at[idx, COL_P])
                    r_n   = parse_br_float(df_ed.at[idx, COL_R])
                    rho_n = parse_br_float(df_ed.at[idx, COL_RHO])
                    r_o   = parse_br_float(df_interno.at[idx, COL_R])   if idx in df_interno.index else None
                    rho_o = parse_br_float(df_interno.at[idx, COL_RHO]) if idx in df_interno.index else None
                    a_o   = parse_br_float(df_interno.at[idx, COL_A])   if idx in df_interno.index else None
                    p_o   = parse_br_float(df_interno.at[idx, COL_P])   if idx in df_interno.index else None
                    if (r_n and a_n and p_n and
                            (check_diff(r_n,r_o) or check_diff(a_n,a_o) or check_diff(p_n,p_o))):
                        c = round(palmer_rho_aparente(a_n, r_n, p_n, correcao_haste), 2)
                        if check_diff(c, rho_n):
                            df_ed.at[idx, COL_RHO] = c; mod = True
                    elif rho_n and a_n and p_n and check_diff(rho_n, rho_o):
                        c = round(palmer_resistencia(a_n, rho_n, p_n, correcao_haste), 4)
                        if check_diff(c, r_n):
                            df_ed.at[idx, COL_R] = c; mod = True
                st.session_state.dados_tabelas[d] = df_ed
                if mod:
                    reseta_calculo(); st.session_state.precisa_atualizar = True

if st.session_state.get("precisa_atualizar"):
    st.session_state.precisa_atualizar = False; st.rerun()

# =========================================================================
# MAPA SATÉLITE — visível por padrão
# =========================================================================
st.divider()
mostrar_mapa = st.checkbox("🗺️ Renderizar Mapa de Satélite", value=True)
if mostrar_mapa:
    try:
        import folium; from folium import plugins; from streamlit_folium import st_folium

        # ══════════════ BARRA DE CONTROLE DO MAPA ══════════════
        # Modo desenho + seletor de eixo ATIVO + ações rápidas
        # (Tudo numa única linha para não ocupar espaço)
        with st.container(border=True):
            col_dm1, col_dm2, col_dm3, col_dm4, col_dm5 = st.columns([1.4, 1.4, 0.7, 0.7, 0.8])

            modo_desenho = col_dm1.checkbox(
                "✏️ Modo Desenho",
                value=st.session_state.get("modo_desenho_persist", False),
                key="modo_desenho_persist",
                help="Ative para desenhar elementos no mapa. As cores seguem o Eixo Ativo.")

            # Seletor de Eixo Ativo (qual eixo ganha cor ao desenhar)
            opcoes_eixo_ativo = list(direcoes_ativas) + ["Outro"]
            eixo_ativo_idx = 0
            if "eixo_ativo_desenho" in st.session_state:
                if st.session_state.eixo_ativo_desenho in opcoes_eixo_ativo:
                    eixo_ativo_idx = opcoes_eixo_ativo.index(st.session_state.eixo_ativo_desenho)
            eixo_ativo = col_dm2.selectbox(
                "🎨 Eixo ativo para desenhar:",
                opcoes_eixo_ativo,
                index=eixo_ativo_idx,
                key="eixo_ativo_desenho",
                disabled=not modo_desenho,
                help="Escolha o eixo ANTES de desenhar — a cor é aplicada automaticamente.")

            # Mostra a cor do eixo ativo (chip visual)
            cor_ativa = COR_EIXO_HEX.get(eixo_ativo, "#3388ff")
            col_dm3.markdown(
                f"<div style='display:flex;align-items:center;height:38px;"
                f"justify-content:center;background:{cor_ativa};color:white;"
                f"border-radius:6px;font-weight:700;font-family:sans-serif;'>"
                f"● Eixo {eixo_ativo}"
                f"</div>",
                unsafe_allow_html=True)

            # Estado dos desenhos do laudo atual
            desenhos_atuais = st.session_state.get("desenhos_geojson", {
                "type": "FeatureCollection", "features": []
            })
            for i, feat in enumerate(desenhos_atuais.get("features", [])):
                feat.setdefault("properties", {})
                feat["properties"].setdefault("_id", f"feat_{i}_{int(time.time()*1000)%100000}")

            n_features = len(desenhos_atuais.get("features", []))
            # Espessura padrão para novos desenhos
            esp_pad = col_dm5.number_input(
                "📏 px", min_value=1, max_value=20, value=4, step=1,
                key="feat_esp_default",
                help="Espessura padrão das novas linhas")
            if col_dm4.button(f"🗑️ Limpar ({n_features})", use_container_width=True,
                              disabled=n_features == 0):
                st.session_state.desenhos_geojson = {"type":"FeatureCollection","features":[]}
                if st.session_state.get("laudo_atual_id"):
                    adicionar_log(st.session_state.laudo_atual_id,
                                  "desenhos_limpos", "Todos os desenhos removidos")
                st.toast("Desenhos limpos", icon="🗑️"); st.rerun()

        # ── Construção do mapa ────────────────────────────────────────────
        m = folium.Map(location=[lat_input, lon_input], zoom_start=15,
                       max_zoom=22, tiles=None)
        folium.TileLayer('https://mt1.google.com/vt/lyrs=y&x={x}&y={y}&z={z}',
                         attr='Google Maps', max_zoom=22, max_native_zoom=20).add_to(m)


        # Marcador central
        folium.CircleMarker([lat_input, lon_input], radius=7, color='yellow',
                            fill=True, fill_opacity=1, tooltip="Centro").add_to(m)
        plugins.LocateControl().add_to(m)

        # ── Re-renderiza desenhos salvos como camadas individuais ────────
        # Cada feature vira uma camada própria para mostrar tooltip rico
        # com Eixo (atribuído pelo usuário) e comprimento.
        for feat in desenhos_atuais.get("features", []):
            props = feat.get("properties", {})
            eixo_atr = props.get("eixo", "—")
            geom_type = feat.get("geometry", {}).get("type", "")
            comp_m = comprimento_feature_m(feat)

            # Cor: prioriza eixo atribuído > cor padrão
            cor = COR_EIXO_HEX.get(eixo_atr, props.get("color", "#3388ff"))

            # Texto do tooltip — DESTAQUE em HTML (sticky=True para ficar visível)
            tip_html = (
                f"<div style='font-family:sans-serif;font-size:13px;line-height:1.5'>"
                f"<b>Eixo:</b> <span style='color:{cor};font-weight:700'>{eixo_atr}</span><br>"
                f"<b>Tipo:</b> {descrever_feature(feat)}<br>"
                f"<b>Comprimento:</b> {comp_m:.1f} m"
                f"</div>"
            ) if eixo_atr != "—" else (
                f"<div style='font-family:sans-serif;font-size:12px;color:#888'>"
                f"<i>Sem eixo atribuído ({descrever_feature(feat)})</i><br>"
                f"<b>Comprimento:</b> {comp_m:.1f} m"
                f"</div>"
            )

            try:
                # Renderiza geometria individual (usa espessura salva pelo usuário)
                _peso = int(props.get("weight", 4))
                folium.GeoJson(
                    feat,
                    style_function=lambda _f, _c=cor, _w=_peso: {
                        "color": _c, "weight": _w,
                        "fillColor": _c, "fillOpacity": 0.25
                    },
                    highlight_function=lambda _f, _c=cor, _w=_peso: {
                        "color": _c, "weight": _w + 3, "fillOpacity": 0.4
                    },
                    tooltip=folium.Tooltip(tip_html, sticky=True),
                ).add_to(m)

                # Adiciona LABEL (rótulo permanente) no centro da feature
                if eixo_atr != "—":
                    coords = feat.get("geometry", {}).get("coordinates", [])
                    centro_ll = None
                    if geom_type == "LineString" and len(coords) >= 2:
                        # Ponto médio
                        ic = len(coords) // 2
                        centro_ll = (coords[ic][1], coords[ic][0])
                    elif geom_type == "Polygon" and coords:
                        ring = coords[0]
                        if ring:
                            avg_ln = sum(c[0] for c in ring) / len(ring)
                            avg_lt = sum(c[1] for c in ring) / len(ring)
                            centro_ll = (avg_lt, avg_ln)
                    elif geom_type == "Point" and len(coords) >= 2:
                        centro_ll = (coords[1], coords[0])

                    if centro_ll:
                        label_html = (
                            f"<div style='background:{cor};color:white;"
                            f"padding:2px 8px;border-radius:3px;font-weight:700;"
                            f"font-size:11px;font-family:sans-serif;"
                            f"box-shadow:0 1px 3px rgba(0,0,0,.4);"
                            f"white-space:nowrap'>"
                            f"Eixo {eixo_atr} · {comp_m:.0f}m"
                            f"</div>"
                        )
                        folium.Marker(
                            location=centro_ll,
                            icon=folium.DivIcon(
                                html=label_html,
                                icon_size=(120, 24),
                                icon_anchor=(60, 12)
                            )
                        ).add_to(m)
            except Exception:
                pass

        # ── Adiciona Draw plugin se modo desenho ativo ────────────────────
        if modo_desenho:
            cor_draw  = COR_EIXO_HEX.get(eixo_ativo, "#3388ff")
            esp_draw  = int(st.session_state.get("feat_esp_default", 4))
            draw = plugins.Draw(
                export=False,
                position='topleft',
                draw_options={
                    'polyline':  {'shapeOptions': {'color': cor_draw, 'weight': esp_draw}},
                    'polygon':   {'shapeOptions': {'color': cor_draw, 'fillColor': cor_draw,
                                                    'fillOpacity': 0.3, 'weight': esp_draw}},
                    'rectangle': {'shapeOptions': {'color': cor_draw, 'fillColor': cor_draw,
                                                    'fillOpacity': 0.25, 'weight': esp_draw}},
                    'circle':    {'shapeOptions': {'color': cor_draw, 'fillColor': cor_draw,
                                                    'fillOpacity': 0.25}},
                    'marker': True,
                    'circlemarker': False,
                },
                edit_options={'edit': True, 'remove': True}
            )
            draw.add_to(m)
            plugins.MeasureControl(
                position='topright', primary_length_unit='meters',
                secondary_length_unit='kilometers',
                primary_area_unit='sqmeters',
                secondary_area_unit='hectares').add_to(m)

        # ── Render ────────────────────────────────────────────────────────
        if modo_desenho:
            # Key muda só quando coordenadas mudam (recentra o mapa)
            # NÃO muda ao desenhar → mapa não some
            _map_key = f"mapa_draw_{round(lat_input,4)}_{round(lon_input,4)}"
            map_data = st_folium(m, height=500, use_container_width=True,
                                 key=_map_key,
                                 returned_objects=["last_active_drawing"])

            # ── Captura APENAS o último desenho feito ──────────────────────
            # last_active_drawing retorna só a feature recém-finalizada.
            # Não varia a cada frame como all_drawings → mapa não pisca.
            novo_feat = map_data.get("last_active_drawing") if map_data else None
            if novo_feat and isinstance(novo_feat, dict) and novo_feat.get("geometry"):
                cor_eixo_atual = COR_EIXO_HEX.get(eixo_ativo, "#3388ff")
                existentes = desenhos_atuais.get("features", [])
                # Evita duplicatas por geometria
                novo_geom = str(novo_feat.get("geometry", ""))
                ja_existe = any(
                    str(f.get("geometry", "")) == novo_geom
                    for f in existentes)
                if not ja_existe:
                    novo_feat.setdefault("properties", {})
                    novo_feat["properties"]["eixo"]   = eixo_ativo
                    novo_feat["properties"]["color"]  = cor_eixo_atual
                    novo_feat["properties"]["weight"] = int(st.session_state.get(
                        f"feat_esp_default", 4))
                    novo_feat["properties"]["_id"] = (
                        f"feat_{len(existentes)}_{int(time.time()*1000)%100000}")
                    st.session_state.desenhos_geojson = {
                        "type": "FeatureCollection",
                        "features": existentes + [novo_feat]
                    }
                    if st.session_state.get("laudo_atual_id"):
                        adicionar_log(st.session_state.laudo_atual_id,
                                      "desenho_adicionado",
                                      f"Eixo {eixo_ativo} | "
                                      f"Total: {len(existentes)+1}")
        else:
            st_folium(m, height=420, use_container_width=True,
                      returned_objects=[], key=f"mapa_main_{round(lat_input,4)}_{round(lon_input,4)}")

        # ── Painel de edição dos desenhos (SEMPRE VISÍVEL) ───────────────
        # Não está mais sob "if n_features > 0" — botão fixo no topo
        with st.container(border=True):
            n_atual = len(desenhos_atuais.get("features", []))
            cabec_col1, cabec_col2 = st.columns([3, 1])
            cabec_col1.markdown(f"### 📋 Edição de Desenhos do Mapa ({n_atual})")
            ver_painel = cabec_col2.toggle(
                "Mostrar painel",
                value=st.session_state.get("ver_painel_desenhos", n_atual > 0),
                key="ver_painel_desenhos",
                help="Mantém o painel de edição sempre acessível"
            )

        if ver_painel and n_atual == 0:
            with st.container(border=True):
                st.info("📭 **Nenhum desenho ainda.** Ative **✏️ Modo Desenho** acima e "
                        "selecione o **🎨 Eixo ativo**. Depois use as ferramentas que "
                        "aparecem no canto superior esquerdo do mapa para desenhar.")

        if ver_painel and n_atual > 0:
            with st.container(border=True):
                st.caption("Atribua ou edite o **eixo** de cada desenho. "
                           "A cor no mapa e o rótulo flutuante atualizam automaticamente.")

                opcoes_eixo = ["—"] + list(direcoes_ativas) + ["Outro"]

                # Cabeçalho da tabela
                hc1, hc2, hc3, hc4, hc5, hc6, hc7 = st.columns([0.3, 1.0, 0.9, 0.7, 0.7, 1.0, 0.5])
                hc1.markdown("**#**")
                hc2.markdown("**Tipo**")
                hc3.markdown("**Eixo**")
                hc4.markdown("**Compr.(m)**")
                hc5.markdown("**Esp.(px)**")
                hc6.markdown("**Observação**")
                hc7.markdown("**🗑️**")
                st.divider()

                houve_alteracao = False
                idx_para_remover = None

                for i, feat in enumerate(desenhos_atuais.get("features", [])):
                    props = feat.setdefault("properties", {})
                    rc1, rc2, rc3, rc4, rc5, rc6, rc7 = st.columns([0.3, 1.0, 0.9, 0.7, 0.7, 1.0, 0.5])
                    rc1.markdown(f"**{i+1}**")
                    rc2.markdown(descrever_feature(feat))

                    eixo_atual = props.get("eixo", "—")
                    if eixo_atual not in opcoes_eixo:
                        eixo_atual = "Outro"
                    novo_eixo = rc3.selectbox(
                        "Eixo", opcoes_eixo,
                        index=opcoes_eixo.index(eixo_atual),
                        key=f"feat_eixo_{props.get('_id', i)}",
                        label_visibility="collapsed")
                    if novo_eixo != props.get("eixo"):
                        props["eixo"] = novo_eixo
                        if novo_eixo in COR_EIXO_HEX:
                            props["color"] = COR_EIXO_HEX[novo_eixo]
                        houve_alteracao = True

                    rc4.markdown(f"`{comprimento_feature_m(feat):.1f}`")

                    # Espessura da linha
                    esp_atual = int(props.get("weight", 4))
                    nova_esp = rc5.number_input(
                        "Esp", value=esp_atual, min_value=1, max_value=20, step=1,
                        key=f"feat_esp_{props.get('_id', i)}",
                        label_visibility="collapsed")
                    if nova_esp != esp_atual:
                        props["weight"] = int(nova_esp)
                        houve_alteracao = True

                    obs_atual = props.get("obs", "")
                    nova_obs = rc6.text_input(
                        "Obs", value=obs_atual,
                        key=f"feat_obs_{props.get('_id', i)}",
                        label_visibility="collapsed",
                        placeholder="opcional")
                    if nova_obs != obs_atual:
                        props["obs"] = nova_obs
                        houve_alteracao = True

                    if rc7.button("🗑️", key=f"feat_del_{props.get('_id', i)}",
                                  help="Remover este desenho"):
                        idx_para_remover = i

                # Aplicar exclusão
                if idx_para_remover is not None:
                    del desenhos_atuais["features"][idx_para_remover]
                    st.session_state.desenhos_geojson = desenhos_atuais
                    if st.session_state.get("laudo_atual_id"):
                        adicionar_log(st.session_state.laudo_atual_id,
                                      "desenho_removido",
                                      f"Item #{idx_para_remover+1}")
                    st.rerun()

                # Persistir alterações
                if houve_alteracao:
                    st.session_state.desenhos_geojson = desenhos_atuais
                    if st.session_state.get("laudo_atual_id"):
                        adicionar_log(st.session_state.laudo_atual_id,
                                      "desenho_editado",
                                      "Eixo/observação atualizado")
                    st.rerun()

                # Resumo por eixo
                if any(f.get("properties", {}).get("eixo", "—") != "—"
                       for f in desenhos_atuais["features"]):
                    st.divider()
                    st.markdown("**📊 Resumo por Eixo:**")
                    resumo = {}
                    for f in desenhos_atuais["features"]:
                        eixo_x = f.get("properties", {}).get("eixo", "—")
                        if eixo_x == "—": continue
                        c = comprimento_feature_m(f)
                        resumo.setdefault(eixo_x, {"qtd": 0, "comp_total": 0.0})
                        resumo[eixo_x]["qtd"] += 1
                        resumo[eixo_x]["comp_total"] += c
                    cols_res = st.columns(min(len(resumo), 6) or 1)
                    for idx, (eixo_x, dados) in enumerate(sorted(resumo.items())):
                        cols_res[idx % len(cols_res)].metric(
                            f"Eixo {eixo_x}",
                            f"{dados['comp_total']:.1f} m",
                            f"{dados['qtd']} item(ns)")

        # ── Exportação GeoJSON / KML ──────────────────────────────────────
        if n_features > 0:
            with st.expander(f"📦 Exportar desenhos ({n_features} feature(s))", expanded=False):
                ce1, ce2 = st.columns(2)
                # GeoJSON
                geojson_bytes = json.dumps(desenhos_atuais, ensure_ascii=False,
                                          indent=2).encode("utf-8")
                ce1.download_button(
                    "📥 GeoJSON (.geojson)", data=geojson_bytes,
                    file_name=f"iluvium_desenhos_{datetime.now().strftime('%Y%m%d_%H%M')}.geojson",
                    mime="application/geo+json", use_container_width=True,
                    help="Abre no QGIS, ArcGIS, geopandas")

                # KML
                def geojson_to_kml(gj):
                    """Conversão simples GeoJSON → KML para uso no Google Earth/QGIS."""
                    kml = ['<?xml version="1.0" encoding="UTF-8"?>',
                           '<kml xmlns="http://www.opengis.net/kml/2.2">',
                           '<Document>',
                           f'<name>Iluvium - Desenhos do Laudo</name>']
                    for i, feat in enumerate(gj.get("features", [])):
                        geom = feat.get("geometry", {})
                        gtype = geom.get("type", "")
                        coords = geom.get("coordinates", [])
                        props = feat.get("properties", {})
                        nome = props.get("name", f"Feature_{i+1}")
                        kml.append(f'<Placemark><name>{nome}</name>')
                        if gtype == "Point":
                            kml.append(f'<Point><coordinates>{coords[0]},{coords[1]}'
                                      '</coordinates></Point>')
                        elif gtype == "LineString":
                            cs = " ".join(f"{c[0]},{c[1]}" for c in coords)
                            kml.append(f'<LineString><coordinates>{cs}'
                                      '</coordinates></LineString>')
                        elif gtype == "Polygon":
                            ring = coords[0] if coords else []
                            cs = " ".join(f"{c[0]},{c[1]}" for c in ring)
                            kml.append(f'<Polygon><outerBoundaryIs><LinearRing>'
                                      f'<coordinates>{cs}</coordinates>'
                                      '</LinearRing></outerBoundaryIs></Polygon>')
                        kml.append('</Placemark>')
                    kml.append('</Document></kml>')
                    return "\n".join(kml).encode("utf-8")

                ce2.download_button(
                    "📥 KML (Google Earth)", data=geojson_to_kml(desenhos_atuais),
                    file_name=f"iluvium_desenhos_{datetime.now().strftime('%Y%m%d_%H%M')}.kml",
                    mime="application/vnd.google-earth.kml+xml",
                    use_container_width=True,
                    help="Abre no Google Earth e QGIS")

    except ImportError:
        st.warning("Instale **folium** e **streamlit-folium** para o mapa satélite:  \n`pip install folium streamlit-folium`")

# =========================================================================
# CURVA MÉDIA ρ vs PROFUNDIDADE WENNER
# (sempre visível, não precisa calcular Stefanescu)
# =========================================================================
st.divider()
st.markdown("### 🌐 Resistividade Média vs Profundidade (Wenner)")
st.caption("A profundidade efetiva de investigação do método Wenner é **z = 0.519 × a** (Roy & Apparao, 1971). "
           "A curva exibe a média de ρ de todos os eixos por profundidade, com faixa min-máx.")

# Coleta dos pontos por eixo (não depende do cálculo Stefanescu)
_eixos_curva = []
for _d in direcoes_ativas:
    try:
        _df = st.session_state.dados_tabelas[_d]
        _df_clean = _df[_df[COL_A].apply(parse_br_float).notna() &
                        _df[COL_RHO].apply(parse_br_float).notna()]
        if not _df_clean.empty:
            _eixos_curva.append(_d)
    except Exception:
        pass

if _eixos_curva:
    # Coleta pares (z_wenner, rho) por eixo
    _pontos_eixo = {}
    for _d in _eixos_curva:
        _df = st.session_state.dados_tabelas[_d]
        _pares = []
        for _, _row in _df.iterrows():
            _av = parse_br_float(_row[COL_A])
            _rv = parse_br_float(_row[COL_RHO])
            if _av and _rv:
                _z = round(0.519 * _av, 2)   # ← profundidade Wenner real
                _pares.append((_z, _rv))
        if _pares:
            _pontos_eixo[_d] = sorted(_pares)

    # Agrupamento por z (tolerância 5%) e cálculo da média
    _todos = [(_z, _r) for _p in _pontos_eixo.values() for (_z, _r) in _p]
    _z_unicos = sorted(set(round(_z, 2) for _z, _ in _todos))
    _curva = []
    for _z_ref in _z_unicos:
        _vals = [_r for _z, _r in _todos if abs(_z - _z_ref) <= max(0.05 * _z_ref, 0.01)]
        if _vals:
            _curva.append({
                "z_m":     _z_ref,
                "a_m":     round(_z_ref / 0.519, 2),
                "rho_med": float(np.mean(_vals)),
                "rho_min": float(min(_vals)),
                "rho_max": float(max(_vals)),
                "sigma":   float(np.std(_vals)) if len(_vals) > 1 else 0.0,
                "n":       len(_vals)
            })

    _col_g, _col_col, _col_m = st.columns([1.5, 0.7, 1])

    # ── Paleta Res2DInv 16 bins (azul→ciano→verde→amarelo→laranja→vermelho) ──
    _PALETA_1D = [
        "#08006B","#0D00C9","#0017FF","#0066FF","#00B4FF",
        "#00FFFF","#7FFFD4","#00FA9A","#00FF00","#9ACD32",
        "#FFFF00","#FFC700","#FF8C00","#FF4500","#FF0000","#8B0000"
    ]

    def _cor_rho(rho, rho_min, rho_max):
        """Mapeia ρ para cor da paleta (escala log)."""
        if rho_max <= rho_min: return _PALETA_1D[8]
        import math
        t = (math.log10(max(rho, 1e-3)) - math.log10(max(rho_min, 1e-3))) / \
            (math.log10(max(rho_max, 1e-3)) - math.log10(max(rho_min, 1e-3)))
        t = max(0.0, min(1.0, t))
        idx = min(int(t * len(_PALETA_1D)), len(_PALETA_1D)-1)
        return _PALETA_1D[idx]

    with _col_g:
        _fig_c = go.Figure()

        if _curva:
            _za  = [p["z_m"]     for p in _curva]
            _rmi = [p["rho_min"] for p in _curva]
            _rma = [p["rho_max"] for p in _curva]
            _rme = [p["rho_med"] for p in _curva]

            # Banda min-máx
            _fig_c.add_trace(go.Scatter(
                x=_rma + _rmi[::-1], y=_za + _za[::-1],
                fill='toself', fillcolor='rgba(150,150,150,0.18)',
                line=dict(color='rgba(0,0,0,0)'), showlegend=True,
                name='Faixa min/máx', hoverinfo='skip'))

            # Curvas individuais (cor do eixo)
            for _d, _pares in _pontos_eixo.items():
                _zd = [p[0] for p in _pares]
                _rd = [p[1] for p in _pares]
                _fig_c.add_trace(go.Scatter(
                    x=_rd, y=_zd, mode='lines+markers',
                    name=f"Eixo {_d}",
                    line=dict(color=COR_MAP[_d], width=2, dash='dot'),
                    marker=dict(size=8, color=COR_MAP[_d],
                                line=dict(width=1, color='white')),
                    hovertemplate=(f'<b>Eixo {_d}</b><br>'
                                   'z=%{y:.2f} m<br>ρ=%{x:.0f} Ω·m<extra></extra>')))

            # Curva média (destaque)
            _fig_c.add_trace(go.Scatter(
                x=_rme, y=_za, mode='lines+markers',
                name='<b>Média</b>',
                line=dict(color='#0a0a0a', width=4),
                marker=dict(size=12, color='#c0392b', symbol='diamond',
                            line=dict(width=2, color='white')),
                hovertemplate='<b>Média</b><br>z=%{y:.2f} m<br>'
                              'ρ médio = %{x:.1f} Ω·m<extra></extra>'))

        _fig_c.update_layout(
            title=dict(text="<b>ρ Média vs Profundidade Wenner</b>",
                       font=dict(size=13), x=0.5),
            xaxis=dict(title="ρ (Ω·m)", type='log',
                       showgrid=True, gridcolor='lightgray'),
            yaxis=dict(title="z = 0.519·a  (m)",
                       autorange='reversed', showgrid=True, gridcolor='lightgray'),
            template='plotly_white', height=480,
            margin=dict(t=45, b=50, l=70, r=10),
            legend=dict(yanchor='bottom', y=0.02, xanchor='right', x=0.98,
                        bgcolor='rgba(255,255,255,0.9)',
                        bordercolor='black', borderwidth=1))
        st.plotly_chart(_fig_c, use_container_width=True)
        st.session_state.fig_pseudo = _fig_c

    # ── Coluna Geoelétrica — estilo NBR/Res2DInv ──────────────────────────
    with _col_col:
        st.markdown(
            "<div style='text-align:center;font-weight:700;font-size:13px;"
            "font-family:sans-serif;margin-bottom:4px;letter-spacing:.05em'>"
            "COLUNA GEOELÉTRICA</div>",
            unsafe_allow_html=True)

        if _curva:
            import math as _math
            _rho_min_all = min(p["rho_min"] for p in _curva)
            _rho_max_all = max(p["rho_max"] for p in _curva)
            _rho_min_log = _math.log10(max(_rho_min_all, 1))
            _rho_max_log = _math.log10(max(_rho_max_all, _rho_min_all * 1.1))

            # Paleta idêntica ao Res2DInv (azul escuro → ciano → verde → amarelo → vermelho)
            _PALETA_NBR = [
                (8,   0, 107), (13,  0, 201), (0,  23, 255), (0, 102, 255),
                (0, 180, 255), (0,  255, 255),(127,255, 212), (0, 250, 154),
                (0, 255,   0), (154,205,  50),(255,255,   0), (255,199,   0),
                (255,140,   0), (255, 69,   0),(255,  0,   0),(139,  0,   0),
            ]

            def _rgb_nbr(rho):
                if _rho_max_log <= _rho_min_log:
                    return _PALETA_NBR[8]
                t = (_math.log10(max(rho, 1)) - _rho_min_log) / (_rho_max_log - _rho_min_log)
                t = max(0.0, min(1.0, t))
                idx = min(int(t * len(_PALETA_NBR)), len(_PALETA_NBR) - 1)
                return _PALETA_NBR[idx]

            def _hex_nbr(rho):
                r, g, b = _rgb_nbr(rho)
                return f"#{r:02X}{g:02X}{b:02X}"

            def _text_color(rho):
                r, g, b = _rgb_nbr(rho)
                lum = 0.299*r + 0.587*g + 0.114*b
                return "black" if lum > 140 else "white"

            _fig_col = go.Figure()

            # Barras coloridas por intervalo de profundidade
            for _k, _p in enumerate(_curva):
                _z_top = 0.0 if _k == 0 else _curva[_k-1]["z_m"]
                _z_bot = _p["z_m"]
                _cor_h = _hex_nbr(_p["rho_med"])
                _tc    = _text_color(_p["rho_med"])

                _fig_col.add_shape(
                    type="rect",
                    x0=0, x1=1,
                    y0=_z_top, y1=_z_bot,
                    fillcolor=_cor_h,
                    line=dict(color="black", width=1),
                    layer="below")

                # Rótulo: valor de ρ + unidade
                _fig_col.add_annotation(
                    x=0.5, y=(_z_top + _z_bot) / 2,
                    text=f"<b>{_p['rho_med']:.0f} Ω·m</b>",
                    showarrow=False,
                    font=dict(size=11, color=_tc, family="Arial"),
                    xanchor="center", yanchor="middle")

            # Barra de cores horizontal (legenda inferior estilo NBR)
            _n = len(_PALETA_NBR)
            for _ci, (_r, _g, _b) in enumerate(_PALETA_NBR):
                _fig_col.add_shape(
                    type="rect",
                    x0=_ci/_n, x1=(_ci+1)/_n,
                    y0=max(p["z_m"] for p in _curva) + 0.15,
                    y1=max(p["z_m"] for p in _curva) + 0.45,
                    fillcolor=f"#{_r:02X}{_g:02X}{_b:02X}",
                    line=dict(color="black", width=0.5),
                    layer="below")

            # Rótulos min/max embaixo da barra de cores
            _z_bar = max(p["z_m"] for p in _curva) + 0.5
            _fig_col.add_annotation(
                x=0, y=_z_bar,
                text=f"<b>{_rho_min_all:.0f}</b>",
                showarrow=False, font=dict(size=9, color="black"),
                xanchor="left", yanchor="top")
            _fig_col.add_annotation(
                x=0.5, y=_z_bar,
                text="<b>Resistividade (Ω·m)</b>",
                showarrow=False, font=dict(size=9, color="black"),
                xanchor="center", yanchor="top")
            _fig_col.add_annotation(
                x=1, y=_z_bar,
                text=f"<b>{_rho_max_all:.0f}</b>",
                showarrow=False, font=dict(size=9, color="black"),
                xanchor="right", yanchor="top")

            # Linha de superfície
            _fig_col.add_shape(
                type="line", x0=0, x1=1, y0=0, y1=0,
                line=dict(color="black", width=2))

            _fig_col.update_layout(
                xaxis=dict(visible=False, range=[0, 1]),
                yaxis=dict(
                    title="Profundidade z (m)",
                    autorange="reversed",
                    showgrid=True, gridcolor="#e0e0e0",
                    zeroline=True, zerolinecolor="black", zerolinewidth=2,
                    tickfont=dict(size=11, family="Arial"),
                    range=[-0.1,
                           max(p["z_m"] for p in _curva) + 0.7]),
                template="plotly_white",
                height=500,
                margin=dict(t=10, b=10, l=60, r=10),
                plot_bgcolor="white",
                paper_bgcolor="white")

            st.plotly_chart(_fig_col, use_container_width=True)
            st.session_state.fig_coluna = _fig_col
        else:
            st.info("Preencha dados para gerar a coluna.")

        # Tabela resumo
        if _curva:
            _df_tab = pd.DataFrame(_curva).rename(columns={
                "z_m":"Prof. z (m)", "a_m":"Espaç. a (m)",
                "rho_med":"ρ médio (Ω·m)", "rho_min":"ρ mín",
                "rho_max":"ρ máx", "sigma":"σ", "n":"Eixos"
            }).round(2)
            st.dataframe(_df_tab, use_container_width=True, hide_index=True)

    with _col_m:
        # Marcador único representativo
        try:
            import folium; from streamlit_folium import st_folium
            _rho_r = float(np.mean([p["rho_med"] for p in _curva])) if _curva else 0.0
            if   _rho_r < 100:   _cor_r, _cls = "#16a085", "Baixa"
            elif _rho_r < 500:   _cor_r, _cls = "#f39c12", "Média"
            elif _rho_r < 2000:  _cor_r, _cls = "#e67e22", "Alta"
            else:                _cor_r, _cls = "#c0392b", "Muito Alta"

            _m3 = folium.Map(location=[lat_input, lon_input], zoom_start=17,
                             max_zoom=22, tiles=None)
            folium.TileLayer('https://mt1.google.com/vt/lyrs=y&x={x}&y={y}&z={z}',
                             attr='Google Maps', max_zoom=22, max_native_zoom=20).add_to(_m3)

            _pop = (f"<div style='font-family:sans-serif;text-align:center'>"
                    f"<b>ρ Representativa</b><br>"
                    f"<span style='font-size:26px;color:{_cor_r};font-weight:700'>"
                    f"{_rho_r:.0f}</span> Ω·m<br><i>{_cls}</i><br>"
                    f"<small>{len(_eixos_curva)} eixo(s) · "
                    f"{len(_curva)} profundidade(s)</small></div>")
            folium.CircleMarker(
                [lat_input, lon_input], radius=24, color=_cor_r,
                fill=True, fill_color=_cor_r, fill_opacity=0.85, weight=4,
                tooltip=f"ρ médio: {_rho_r:.0f} Ω·m",
                popup=folium.Popup(_pop, max_width=240)).add_to(_m3)
            folium.CircleMarker([lat_input, lon_input], radius=4,
                                color='black', fill=True, fill_opacity=1).add_to(_m3)

            st.markdown(
                f"<div style='background:{_cor_r};color:white;padding:10px;"
                f"border-radius:6px;text-align:center;font-family:sans-serif;"
                f"margin-bottom:8px;box-shadow:0 2px 4px rgba(0,0,0,.2)'>"
                f"<div style='font-size:11px;letter-spacing:.1em;opacity:.9;"
                f"text-transform:uppercase'>Resistividade Representativa</div>"
                f"<div style='font-size:34px;font-weight:700;line-height:1.2'>"
                f"{_rho_r:.0f}</div>"
                f"<div style='font-size:13px;opacity:.95'>Ω·m · {_cls}</div>"
                f"</div>", unsafe_allow_html=True)
            st_folium(_m3, height=360, use_container_width=True,
                      returned_objects=[], key="mapa_repr")
        except ImportError:
            st.info("Instale folium para o mapa representativo.")

# Aviso quando sem dados
if not _eixos_curva:
    st.info("💡 Preencha dados de resistividade em pelo menos 1 eixo para visualizar a curva.")

# =========================================================================
# ███ MOTOR DE CÁLCULO ███
# =========================================================================
st.divider()
st.markdown("### 🔬 Motor de Estratificação Geofísica")

col_nc, col_info, col_btn = st.columns([1, 2, 2])
n_cam = col_nc.selectbox("Camadas:", [2, 3, 4], index=1, on_change=reseta_calculo,
    help="2=simples | 3=padrão NBR 7117 | 4=solos complexos")
_pinfo = {2:"3 parâm. (ρ₁,ρ₂,h₁)", 3:"5 parâm. (ρ₁,ρ₂,ρ₃,h₁,h₂)",
          4:"7 parâm. (ρ₁,ρ₂,ρ₃,ρ₄,h₁,h₂,h₃)"}
col_info.info(f"**Modelo {n_cam} camadas** — {_pinfo[n_cam]}")
btn_calc = col_btn.button("⚡ INICIAR CÁLCULO E OTIMIZAÇÃO",
                          type="primary", use_container_width=True)

if btn_calc:
    reseta_calculo()
    _, _, differential_evolution = load_scipy()

    a_g, rho_g, dir_g = [], [], []
    for d in direcoes_ativas:
        df_v = st.session_state.dados_tabelas[d].copy()
        df_v["_a"]   = df_v[COL_A].apply(parse_br_float)
        df_v["_rho"] = df_v[COL_RHO].apply(parse_br_float)
        df_v = df_v.dropna(subset=["_a","_rho"])
        a_g.extend(df_v["_a"].tolist()); rho_g.extend(df_v["_rho"].tolist())
        dir_g.extend([d]*len(df_v))
    a_g = np.array(a_g, dtype=float); rho_g = np.array(rho_g, dtype=float)

    if len(a_g) < n_cam + 1:
        st.error(f"Mínimo {n_cam+1} pontos para {n_cam} camadas. Você tem {len(a_g)}.")
        st.stop()

    rho_min = max(0.5, float(np.min(rho_g))*0.02)
    rho_max = float(np.max(rho_g))*80.0
    a_max   = float(np.max(a_g))
    b_r = (rho_min, rho_max)
    b_h = [(0.05, min(a_max*0.8,  30.0)),
           (0.05, min(a_max*1.5,  60.0)),
           (0.05, min(a_max*3.0, 120.0))]

    if   n_cam == 2: bounds = [b_r, b_r, b_h[0]]
    elif n_cam == 3: bounds = [b_r, b_r, b_r, b_h[0], b_h[1]]
    else:            bounds = [b_r, b_r, b_r, b_r, b_h[0], b_h[1], b_h[2]]

    def penalizacao(x, nc):
        pen = 0.0; rs = x[:nc]; hs = x[nc:]
        for j in range(nc-1):
            r = rs[j+1] / max(rs[j], 1e-3)
            if r > 100: pen += (r-100)*0.005
            if r < 0.01: pen += (0.01-r)*10
        for h in hs:
            if h < 0.05: pen += (0.05-h)*5
        return pen

    def objetivo(x, nc, av, rv):
        yc = np.clip(modelo_fast(av, x, nc), 1e-3, None)
        return np.sqrt(np.mean((np.log(yc)-np.log(rv))**2)) + penalizacao(x, nc)

    _de = {2:dict(popsize=12,maxiter=120,mutation=(0.5,1.0),recombination=0.70),
           3:dict(popsize=14,maxiter=160,mutation=(0.5,1.0),recombination=0.75),
           4:dict(popsize=16,maxiter=220,mutation=(0.5,1.0),recombination=0.80)}

    prog = st.progress(0, text="🔄 Etapa 1/2: Otimização global (Differential Evolution)…")
    t0 = time.time()
    with st.spinner(f"Otimizando modelo de {n_cam} camadas…"):
        res = differential_evolution(objetivo, bounds, args=(n_cam, a_g, rho_g),
                                     seed=42, polish=True, tol=1e-6, workers=1, **_de[n_cam])
    dt = time.time() - t0
    prog.progress(70, text="🔄 Etapa 2/2: Refinamento exato (Hankel adaptativo)…")

    rho_calc = np.clip(modelo_exact(a_g, res.x, n_cam), 1e-3, None)
    rms_log  = float(np.sqrt(np.mean((np.log(rho_calc)-np.log(rho_g))**2)))
    rms_pct  = float(np.sqrt(np.mean(((rho_calc-rho_g)/rho_g)**2))*100.0)

    prog.progress(100, text=f"✅ Concluído em {dt:.1f}s — Erro RMS: {rms_pct:.2f}%")
    time.sleep(0.3); prog.empty()

    # Monta df_r e df_d AQUI e salva no session_state para o botão PDF
    rhos_r = list(res.x[:n_cam]); hs_r = list(res.x[n_cam:])
    _suf = ["(Sup.)"] + [f"(Int.{j})" for j in range(1,n_cam-1)] + ["(Prof.)"]
    _df_r = pd.DataFrame({
        "Camada":  [f"Cam. {j+1} {_suf[j]}" for j in range(n_cam)],
        "rho":     [f"{r:.1f} Ohm.m" for r in rhos_r],
        "h":       [f"{h:.2f} m" for h in hs_r] + ["Infinita"]
    })
    dv_r = (rho_calc - rho_g) / rho_g * 100.0
    _df_d = pd.DataFrame({
        "a [m]":    np.round(a_g, 2),
        "Eixo":     dir_g,
        "Medida":   np.round(rho_g, 1),
        "Calculada":np.round(rho_calc, 1),
        "Desvio %": np.round(dv_r, 2)
    })

    st.session_state.update(dict(
        res_x=res.x, camadas_atuais=n_cam, a_g=a_g, rho_g=rho_g, dir_g=dir_g,
        rho_final_calc=rho_calc, erro_rms_log=rms_log, erro_rms_pct=rms_pct,
        lat_center=lat_input, lon_center=lon_input,
        df_r_resultado=_df_r, df_d_resultado=_df_d,
        calc_concluido=True))

    # Log no laudo aberto (se houver)
    if st.session_state.get("laudo_atual_id"):
        adicionar_log(
            st.session_state.laudo_atual_id,
            "calculo_stefanescu",
            f"{n_cam} camadas | RMS={rms_pct:.3f}% | RMSlog={rms_log:.4f}")

# Guard — bloqueia só os resultados técnicos do Stefanescu
if not st.session_state.get('calc_concluido', False):
    st.info("ℹ️ Configure os dados e clique em **⚡ INICIAR CÁLCULO** para ver os resultados de estratificação.")

    # Botões sempre visíveis mesmo sem cálculo
    st.divider()
    _col_pdf0, _col_chat0 = st.columns(2)
    with _col_pdf0:
        _pdf_ok = (st.session_state.get('df_r_resultado') is not None)
        if _pdf_ok:
            if st.button("📄 Gerar Laudo PDF Iluvium Pro", type="primary",
                         use_container_width=True):
                modal_pdf(
                    st.session_state.df_r_resultado,
                    st.session_state.df_d_resultado,
                    st.session_state.erro_rms_log,
                    st.session_state.erro_rms_pct,
                    lat_input,
                    lon_input,
                    st.session_state.get('fig_resultado'),
                    fig_ps=st.session_state.get('fig_pseudo'),
                    desenhos_geojson=st.session_state.get('desenhos_geojson'))
        else:
            st.button("📄 Gerar Laudo PDF Iluvium Pro", type="primary",
                      use_container_width=True, disabled=True,
                      help="Execute o cálculo primeiro para gerar o laudo completo")
    with _col_chat0:
        if st.button("🤖 Abrir Chat Assistente IA", use_container_width=True,
                     help="Tire dúvidas sobre NBR 7117, IEEE 80, malhas de aterramento e mais"):
            if not st.session_state.chat_api_key:
                st.warning("Configure a API Key na sidebar primeiro!")
            else:
                st.session_state.chat_aberto = True
                st.rerun()
    st.stop()

# =========================================================================
# RESULTADOS (só renderiza após cálculo Stefanescu)
# =========================================================================

a_g     = st.session_state.a_g
rho_g   = st.session_state.rho_g
dir_g   = st.session_state.dir_g
res_x   = st.session_state.res_x
rms_log = st.session_state.erro_rms_log
rms_pct = st.session_state.erro_rms_pct
calc_o  = st.session_state.rho_final_calc
nc      = st.session_state.camadas_atuais
rhos    = list(res_x[:nc])
hs      = list(res_x[nc:])
df_r    = st.session_state.df_r_resultado
df_d    = st.session_state.df_d_resultado

st.divider()
st.subheader(f"📊 Resultados | {nc} Camadas")

if   rms_pct < 5:  st.success(f"✅ Ajuste EXCELENTE — RMS = {rms_pct:.2f}%  (< 5%)")
elif rms_pct < 10: st.info(f"👍 Ajuste BOM — RMS = {rms_pct:.2f}%  (5–10%)")
elif rms_pct < 20: st.warning(f"⚠️ Ajuste REGULAR — RMS = {rms_pct:.2f}%  (10–20%) — tente mais camadas")
else:              st.error(f"❌ Ajuste RUIM — RMS = {rms_pct:.2f}%  (> 20%) — verifique os dados")

col_g, col_s = st.columns([3.2, 1.5])

with col_g:
    f = go.Figure()
    visto = set()
    for idx in range(len(a_g)):
        d = dir_g[idx]; novo = d not in visto; visto.add(d)
        f.add_trace(go.Scatter(
            x=[a_g[idx]], y=[rho_g[idx]], mode='markers',
            marker=dict(size=13, color=COR_MAP[d], line=dict(width=1.5, color='black')),
            name=f"Eixo {d} (medido)", showlegend=novo, legendgroup=d,
            hovertemplate=f"<b>Eixo {d}</b><br>a=%{{x:.2f}}m<br>rho=%{{y:.1f}} Ohm.m<extra></extra>"))

    ap = np.logspace(np.log10(max(min(a_g)*0.5, 0.1)), np.log10(max(a_g)*1.5), 90)
    yc = modelo_exact(ap, res_x, nc)
    f.add_trace(go.Scatter(x=ap, y=yc, mode='lines',
        line=dict(color='black', width=3.5), name='Ajuste (Hankel exato)'))

    if step_plot_ativo:
        xs, ys = [min(a_g)*0.3], [rhos[0]]; depth = 0.0
        for j in range(nc-1):
            depth += hs[j]; xs += [depth, depth]; ys += [rhos[j], rhos[j+1]]
        xs.append(max(a_g)*2); ys.append(rhos[-1])
        f.add_trace(go.Scatter(x=xs, y=ys, mode='lines',
            line=dict(color='#1f77b4', width=2, dash='dashdot'), name='Step Plot'))

    f.update_layout(
        title=f"Estratificação Geofísica | {nc} Camadas | RMS = {rms_pct:.2f}%",
        xaxis_title="Espaçamento / Profundidade (m)",
        yaxis_title="Resistividade Aparente (Ohm.m)",
        xaxis_type="log", yaxis_type="log",
        template="plotly_white", height=490,
        legend=dict(orientation="h", yanchor="top", y=-0.16),
        margin=dict(b=90))
    st.plotly_chart(f, use_container_width=True)
    st.session_state.fig_resultado = f

with col_s:
    st.markdown("**📋 Estratificação do Solo**")
    st.dataframe(df_r[["Camada","rho","h"]].rename(
        columns={"rho":"Resistividade","h":"Espessura"}),
        hide_index=True, use_container_width=True)

    fs = go.Figure()
    for j, r_j in enumerate(rhos):
        hl = f"{hs[j]:.2f}m" if j<len(hs) else "inf"
        fs.add_trace(go.Bar(x=['Solo'], y=[1], name=f"C{j+1}",
            marker_color=CORES_SOLO[j%4],
            text=f"<b>C{j+1}</b><br>{hl}<br>{r_j:.0f}",
            textposition='inside', insidetextanchor='middle'))
    fs.update_layout(barmode='stack', yaxis=dict(showticklabels=False, autorange="reversed"),
                     height=200+30*nc, showlegend=False, template="plotly_white",
                     margin=dict(t=5,b=5,l=5,r=5))
    st.plotly_chart(fs, use_container_width=True)

    dv = df_d["Desvio %"].values
    def _cor(v):
        a = abs(v)
        if a<=5:  return 'color:#1b5e20;font-weight:bold'
        if a<=10: return 'color:#2e7d32;font-weight:bold'
        if a<=20: return 'color:#e65100;font-weight:bold'
        return 'color:#b71c1c;font-weight:bold'
    styled = (df_d.style.map(_cor, subset=['Desvio %'])
              .format({"a [m]":"{:.1f}","Medida":"{:.1f}",
                       "Calculada":"{:.1f}","Desvio %":"{:+.2f}%"}))
    st.markdown("**📐 Desvios por Ponto**")
    st.dataframe(styled, hide_index=True, use_container_width=True)

# ███ BOTÕES FINAIS ███
# =========================================================================
st.divider()
col_pdf, col_chat_btn = st.columns(2)

with col_pdf:
    if st.button("📄 Gerar Laudo PDF Iluvium Pro", type="primary", use_container_width=True):
        modal_pdf(
            st.session_state.df_r_resultado,
            st.session_state.df_d_resultado,
            st.session_state.erro_rms_log,
            st.session_state.erro_rms_pct,
            lat_input,
            lon_input,
            st.session_state.get('fig_resultado', f),
            fig_ps=st.session_state.get('fig_pseudo', None),
            desenhos_geojson=st.session_state.get('desenhos_geojson', None))

with col_chat_btn:
    if st.button("🤖 Abrir Chat Assistente IA", use_container_width=True,
                 help="Tire dúvidas sobre NBR 7117, IEEE 80, malhas de aterramento e mais"):
        if not st.session_state.chat_api_key:
            st.warning("Configure a API Key na sidebar primeiro!")
        else:
            st.session_state.chat_aberto = True
            st.rerun()
