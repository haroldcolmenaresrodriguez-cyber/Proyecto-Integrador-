import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller
from scipy.stats import kurtosis, chi2_contingency

st.set_page_config(page_title="Monitor de Riesgo Bursátil - Los Endeudados", layout="wide")

st.title("Monitor de Riesgo Bursátil: Choques Externos sobre Ecopetrol ADR")
st.markdown("**Equipo:** Los Endeudados | **Activo Proxy:** Ecopetrol ADR (`EC` - NYSE) | **Periodo:** 2019–2021")


def limpiar_numero_serie(serie):
    if pd.api.types.is_numeric_dtype(serie):
        return pd.to_numeric(serie, errors="coerce")

    s = (
        serie.astype(str)
        .str.replace("\u00a0", "", regex=False)
        .str.replace(" ", "", regex=False)
        .str.strip()
    )

    # Caso 1: tiene punto y coma (ej. 1.234,56) -> quitar puntos de miles y cambiar coma por punto
    ambos = s.str.contains(r"\.", regex=True) & s.str.contains(",", regex=True)
    s.loc[ambos] = (
        s.loc[ambos]
        .str.replace(".", "", regex=False)
        .str.replace(",", ".", regex=False)
    )

    # Caso 2: solo tiene coma (ej. 164,6 o 1234,56) -> cambiar coma por punto
    solo_coma = ~ambos & s.str.contains(",", regex=True)
    s.loc[solo_coma] = s.loc[solo_coma].str.replace(",", ".", regex=False)

    # Caso 3: solo tiene puntos y sin coma (ej. 32.450.000) -> quitar puntos de miles
    solo_punto = ~ambos & ~solo_coma & s.str.contains(r"\.", regex=True)
    s.loc[solo_punto] = s.loc[solo_punto].str.replace(".", "", regex=False)

    return pd.to_numeric(s, errors="coerce")

@st.cache_data
def cargar_datos():
    ruta = "base_precios_depurada_2019_2021_LosEndeudados_v2.csv"
    df = pd.read_csv(ruta, sep=None, engine="python")
    df.columns = [str(c).replace("\ufeff", "").strip() for c in df.columns]

    col_fecha = next((c for c in df.columns if c.lower() in ["fecha", "date"]), df.columns[0])
    df[col_fecha] = pd.to_datetime(df[col_fecha], format="mixed", dayfirst=True, errors="coerce")
    df = df.dropna(subset=[col_fecha]).sort_values(col_fecha).set_index(col_fecha)

    cols_precios = ["EC_ADR_USD_Cierre", "Brent_USD_Cierre", "USD_COP_Cierre"]
    faltantes = [c for c in cols_precios if c not in df.columns]
    if faltantes:
        return pd.DataFrame(), pd.DataFrame(), faltantes, list(df.columns), df.head(3)

    precios = pd.DataFrame(index=df.index)
    for c in cols_precios:
        precios[c] = limpiar_numero_serie(df[c])

    precios = precios.dropna()
    if len(precios) < 2:
        return precios, pd.DataFrame(), [], list(df.columns), df.head(3)

    rendimientos = 100.0 * np.log(precios / precios.shift(1)).dropna()
    rendimientos.columns = ["EC_ADR_ret", "Brent_ret", "USD_COP_ret"]
    return precios, rendimientos, [], list(df.columns), precios.head(3)


precios, rendimientos, cols_faltantes, cols_csv, muestra = cargar_datos()

if rendimientos.empty:
    st.error("No se pudieron calcular rendimientos válidos desde el CSV.")
    if cols_faltantes:
        st.warning(f"Columnas esperadas no encontradas: {cols_faltantes}")
    st.write("Columnas detectadas en el CSV:", cols_csv)
    st.dataframe(muestra)
    st.stop()

st.sidebar.header("Parámetros del Monitor")
rezagos_var = st.sidebar.selectbox("Rezagos del modelo VAR (p):", [1, 3, 6], index=2)
CAPITAL_INICIAL = st.sidebar.number_input("Capital Inicial (COP):", min_value=1_000_000, value=10_000_000, step=1_000_000)
HORIZONTE = st.sidebar.selectbox("Horizonte de simulación (días):", [21, 63, 252], index=0)
N_SIMULACIONES = st.sidebar.selectbox("Número de simulaciones:", [5000, 10000, 20000], index=1)

tab0, tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Inicio",
    "1. Datos y Estacionariedad",
    "2. Modelo VAR",
    "3. Cadenas de Markov",
    "4. Simulación Monte Carlo (COP)",
    "5. Conclusiones"
])

with tab0:
    st.subheader("Problema, pregunta, usuario y decisión")
    st.markdown(
        "**Problema.** El ADR de Ecopetrol cotiza en dólares en la Bolsa de Nueva York y depende del precio "
        "del petróleo, pero un inversionista colombiano mide su riqueza en pesos. Entre 2019 y 2021, y sobre "
        "todo en marzo de 2020, el activo, el Brent y la tasa de cambio tuvieron movimientos extremos. Las "
        "correlaciones del mismo día no muestran si un choque del petróleo o del dólar anticipa los rendimientos "
        "de los días siguientes, cuánto dura una fase adversa ni qué pérdida en pesos puede ocurrir en un mes."
    )
    st.markdown(
        "**Pregunta central.** ¿Cómo se transmiten los choques del Brent y de la tasa de cambio USD/COP sobre "
        "los rendimientos de Ecopetrol ADR entre 2019 y 2021, con qué probabilidad transita el activo entre "
        "regímenes adverso, neutral y favorable, y qué pérdida extrema en pesos enfrentaría una inversión a "
        "21 días hábiles?"
    )
    c_u, c_d = st.columns(2)
    c_u.info("**Usuario:** analista de riesgo de un portafolio en pesos con posición en Ecopetrol ADR.")
    c_d.success("**Decisión:** asignar una reserva de liquidez igual al Expected Shortfall al 99 % de la pérdida simulada.")
    st.markdown("**Flujo del análisis:** Datos diarios → VAR (transmisión de choques) → Estados de Markov → "
                "Simulación Monte Carlo en pesos → Reserva propuesta.")
    st.markdown(
        "**Datos.** Precios de cierre diarios de Ecopetrol ADR (EC, USD por acción), petróleo Brent (BZ=F, USD por "
        "barril) y tasa de cambio (COP=X, COP por USD), descargados de Yahoo Finance con la librería `yfinance` "
        "(ver `PROYECTO.ipynb`). Periodo común: 2 de enero de 2019 a 30 de diciembre de 2021; "
        f"{len(precios)} precios y {len(rendimientos)} rendimientos logarítmicos por serie, sin datos faltantes."
    )
    st.caption("Use la barra lateral para cambiar los rezagos del VAR, el capital, el horizonte y el número de simulaciones.")

with tab1:
    st.subheader("Exploración de Precios y Rendimientos Logarítmicos (%)")
    vista = st.radio("Seleccionar serie a graficar:", ["Rendimientos Logarítmicos (%)", "Precios de Cierre"], horizontal=True)
    if vista == "Rendimientos Logarítmicos (%)":
        st.line_chart(rendimientos)
    else:
        st.line_chart(precios)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Estadísticas Descriptivas de Rendimientos (%)**")
        st.dataframe(rendimientos.describe().round(4))
    with col2:
        st.markdown("**Matriz de Correlaciones Contemporáneas**")
        st.dataframe(rendimientos.corr().round(4))

    st.markdown("**Prueba de Estacionariedad (Dickey-Fuller Aumentada - ADF)**")
    adf_lista = []
    for col in rendimientos.columns:
        res = adfuller(rendimientos[col])
        adf_lista.append({
            "Variable": col,
            "Estadístico ADF": round(res[0], 4),
            "Valor p": round(res[1], 6),
            "Conclusión (5%)": "Estacionaria I(0)" if res[1] < 0.05 else "No Estacionaria"
        })
    st.dataframe(pd.DataFrame(adf_lista).set_index("Variable"))

with tab2:
    st.subheader(f"Estimación del Modelo Vector Autorregresivo - VAR({rezagos_var})")
    modelo = VAR(rendimientos[["EC_ADR_ret", "Brent_ret", "USD_COP_ret"]])
    res_var = modelo.fit(rezagos_var, trend="c")

    prueba_auto = res_var.test_whiteness(nlags=max(10, rezagos_var + 1), adjusted=True)
    prueba_norm = res_var.test_normality()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("¿VAR Estable?", "Sí (Estable)" if res_var.is_stable() else "No")
    c2.metric("Autocorrelación (p-valor)", f"{prueba_auto.pvalue:.4f}")
    c3.metric("Normalidad (p-valor)", f"{prueba_norm.pvalue:.4f}")
    c4.metric("Criterio AIC", f"{res_var.aic:.4f}")

    st.markdown("**Funciones de Impulso-Respuesta Ortogonalizadas (10 días)**")
    irf = res_var.irf(10)
    fig_irf = irf.plot(orth=True)
    st.pyplot(fig_irf)
    plt.close(fig_irf)

    st.markdown("**Pronóstico del VAR a 10 días (rendimientos logarítmicos, %)**")
    try:
        datos_var = rendimientos[["EC_ADR_ret", "Brent_ret", "USD_COP_ret"]].to_numpy()
        punto, inferior, superior = res_var.forecast_interval(datos_var[-res_var.k_ar:], steps=10, alpha=0.05)
        pronostico = pd.DataFrame({
            "EC pronóstico": punto[:, 0],
            "EC límite inferior 95%": inferior[:, 0],
            "EC límite superior 95%": superior[:, 0],
            "Brent pronóstico": punto[:, 1],
            "USD/COP pronóstico": punto[:, 2],
        }, index=pd.RangeIndex(1, 11, name="Día"))
        st.dataframe(pronostico.round(4))
        st.caption("El pronóstico puntual converge rápido a la media histórica (cercana a cero) y el intervalo es amplio: "
                   "el VAR sirve más para medir la transmisión de choques que para predecir el nivel del rendimiento.")
    except Exception as e:
        st.warning(f"No fue posible calcular el pronóstico: {e}")

    with st.expander("Ver Resumen Estadístico Completo del VAR"):
        st.text(res_var.summary())

serie_ec = rendimientos["EC_ADR_ret"]
limite_inferior = float(serie_ec.quantile(1 / 3))
limite_superior = float(serie_ec.quantile(2 / 3))
orden_estados = ["Adverso", "Neutral", "Favorable"]

condiciones = [
    serie_ec <= limite_inferior,
    (serie_ec > limite_inferior) & (serie_ec <= limite_superior),
    serie_ec > limite_superior
]
estados_serie = np.select(condiciones, orden_estados, default="Neutral")
rendimientos_mk = rendimientos.copy()
rendimientos_mk["Estado"] = estados_serie

transiciones = pd.crosstab(
    rendimientos_mk["Estado"].iloc[:-1].to_numpy(),
    rendimientos_mk["Estado"].iloc[1:].to_numpy(),
    rownames=["Estado actual"],
    colnames=["Estado siguiente"]
).reindex(index=orden_estados, columns=orden_estados, fill_value=0)

matriz_markov = transiciones.div(transiciones.sum(axis=1), axis=0)
P = matriz_markov.to_numpy()

with tab3:
    st.subheader("Cadenas de Markov y Regímenes de Mercado")

    valores_eig, vectores_eig = np.linalg.eig(P.T)
    idx_eig = np.argmin(np.abs(valores_eig - 1.0))
    vec_est = np.real(vectores_eig[:, idx_eig])
    dist_estacionaria = pd.Series(vec_est / vec_est.sum(), index=orden_estados)

    col_m1, col_m2 = st.columns(2)
    with col_m1:
        st.markdown(f"**Límite Adverso:** $\le {limite_inferior:.4f}\%$ | **Límite Favorable:** $> {limite_superior:.4f}\%$")
        st.markdown("**Matriz de Probabilidades de Transición ($P$)**")
        st.dataframe(matriz_markov.round(4))

        duracion = {e: 1 / (1 - matriz_markov.loc[e, e]) for e in orden_estados}
        tabla_mk = pd.DataFrame({
            "Frecuencia Histórica (días)": rendimientos_mk["Estado"].value_counts().reindex(orden_estados),
            "Duración Esperada (días)": pd.Series(duracion).round(2),
            "Probabilidad Estacionaria": dist_estacionaria.round(4)
        })
        st.markdown("**Duración Esperada y Distribución Estacionaria**")
        st.dataframe(tabla_mk)

    with col_m2:
        fig_mk, ax = plt.subplots(figsize=(5.5, 4.5))
        im = ax.imshow(P, cmap="Blues", vmin=0, vmax=1)
        plt.colorbar(im, ax=ax, label="Probabilidad")
        ax.set_xticks(range(3))
        ax.set_xticklabels(orden_estados)
        ax.set_yticks(range(3))
        ax.set_yticklabels(orden_estados)
        ax.set_xlabel("Estado siguiente")
        ax.set_ylabel("Estado actual")
        ax.set_title("Matriz de transición de Markov")
        for i in range(3):
            for j in range(3):
                val = matriz_markov.iloc[i, j]
                ax.text(
                    j, i, f"{val:.3f}", ha="center", va="center",
                    color="white" if val > 0.5 else "black", fontweight="bold"
                )
        plt.tight_layout()
        st.pyplot(fig_mk)
        plt.close(fig_mk)

with tab4:
    st.subheader("Simulación Monte Carlo: Markov-Switching Historical Bootstrap")

    columnas_retornos = ["EC_ADR_ret", "Brent_ret", "USD_COP_ret"]
    grupos_historicos = {
        est: rendimientos_mk.loc[rendimientos_mk["Estado"] == est, columnas_retornos].to_numpy()
        for est in orden_estados
    }

    estado_inicial = rendimientos_mk["Estado"].iloc[-1]
    estado_actual_inicial = orden_estados.index(estado_inicial)

    rng = np.random.default_rng(42)
    resultados = []
    trayectorias_estados = []

    for simulacion in range(N_SIMULACIONES):
        estado_actual = estado_actual_inicial
        suma_ec = 0.0
        suma_brent = 0.0
        suma_usd_cop = 0.0
        suma_portafolio = 0.0
        secuencia_sim = [estado_actual]

        for _ in range(HORIZONTE):
            estado_siguiente = rng.choice(len(orden_estados), p=P[estado_actual])
            nombre_estado = orden_estados[estado_siguiente]

            grupo = grupos_historicos[nombre_estado]
            indice = rng.integers(0, len(grupo))
            retorno_ec, retorno_brent, retorno_cambio = grupo[indice]

            suma_ec += retorno_ec
            suma_brent += retorno_brent
            suma_usd_cop += retorno_cambio

            retorno_cop = retorno_ec + retorno_cambio
            suma_portafolio += retorno_cop

            estado_actual = estado_siguiente
            secuencia_sim.append(estado_actual)

        valor_final = CAPITAL_INICIAL * np.exp(suma_portafolio / 100.0)
        perdida = CAPITAL_INICIAL - valor_final

        resultados.append({
            "Simulacion": simulacion + 1,
            "Log_Ret_EC_pct": suma_ec,
            "Log_Ret_Brent_pct": suma_brent,
            "Log_Ret_USD_COP_pct": suma_usd_cop,
            "Log_Ret_Portafolio_COP_pct": suma_portafolio,
            "Valor_Final_COP": valor_final,
            "Perdida_COP": perdida
        })
        trayectorias_estados.append(secuencia_sim)

    resultados_mc = pd.DataFrame(resultados)
    perdidas = resultados_mc["Perdida_COP"]
    kurtosis_val = kurtosis(perdidas, fisher=True)

    def calcular_riesgo(serie_perdidas, confianza):
        var = serie_perdidas.quantile(confianza)
        es = serie_perdidas[serie_perdidas >= var].mean()
        return var, es

    var_95, es_95 = calcular_riesgo(perdidas, 0.95)
    var_99, es_99 = calcular_riesgo(perdidas, 0.99)
    reserva_recomendada = max(0.0, es_99)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("VaR 95%", f"${var_95:,.0f} COP")
    m2.metric("Expected Shortfall 95%", f"${es_95:,.0f} COP")
    m3.metric("VaR 99%", f"${var_99:,.0f} COP")
    m4.metric("Reserva Propuesta (ES 99%)", f"${reserva_recomendada:,.0f} COP")

    st.info(f"**Validación Cuantitativa:** Curtosis excedente de la distribución de pérdidas simuladas: **{kurtosis_val:.2f}** (evidencia cuantitativa de leptocurtosis / colas pesadas).")

    col_t1, col_t2 = st.columns(2)
    with col_t1:
        st.markdown("**Tabla de Indicadores de Riesgo**")
        tabla_riesgo = pd.DataFrame({
            "Indicador": ["VaR 95%", "Expected Shortfall 95%", "VaR 99%", "Expected Shortfall 99%"],
            "Valor_COP": [f"${var_95:,.0f}", f"${es_95:,.0f}", f"${var_99:,.0f}", f"${es_99:,.0f}"]
        })
        st.dataframe(tabla_riesgo.set_index("Indicador"), width="stretch")

    with col_t2:
        st.markdown("**Resumen General de la Simulación**")
        resumen = pd.DataFrame({
            "Indicador": [
                "Capital inicial",
                "Valor final promedio",
                "Valor final mínimo",
                "Valor final máximo",
                "Pérdida promedio",
                "Mediana de pérdidas",
                "Probabilidad de pérdida"
            ],
            "Valor": [
                f"${CAPITAL_INICIAL:,.0f} COP",
                f"${resultados_mc['Valor_Final_COP'].mean():,.0f} COP",
                f"${resultados_mc['Valor_Final_COP'].min():,.0f} COP",
                f"${resultados_mc['Valor_Final_COP'].max():,.0f} COP",
                f"${perdidas.mean():,.0f} COP",
                f"${perdidas.median():,.0f} COP",
                f"{(perdidas > 0).mean() * 100:.2f}%"
            ]
        })
        st.dataframe(resumen.set_index("Indicador"), width="stretch")

    st.markdown("**Distribución Simulada de Pérdidas en COP**")
    fig_hist, ax_h = plt.subplots(figsize=(10, 4.5))
    ax_h.hist(perdidas, bins=60, color="#3498db", edgecolor="black", alpha=0.75)
    ax_h.axvline(var_95, color="orange", linestyle="--", linewidth=2, label=f"VaR 95%: ${var_95:,.0f}")
    ax_h.axvline(var_99, color="red", linestyle="--", linewidth=2, label=f"VaR 99%: ${var_99:,.0f}")
    ax_h.axvline(es_99, color="darkred", linestyle=":", linewidth=2.5, label=f"ES 99%: ${es_99:,.0f}")
    ax_h.set_title("Distribución simulada de pérdidas en COP (Positivo = Pérdida, Negativo = Ganancia)")
    ax_h.set_xlabel("Pérdida o ganancia (COP)")
    ax_h.set_ylabel("Frecuencia")
    ax_h.legend()
    ax_h.grid(alpha=0.25)
    plt.tight_layout()
    st.pyplot(fig_hist)
    plt.close(fig_hist)

    with st.expander("🔍 Auditoría de Trazabilidad de Estados en Escenarios Extremos"):
        st.markdown("Muestra cuántos días transcurrieron en cada régimen de Markov dentro de las 5 trayectorias de mayor pérdida.")
        peores_indices = resultados_mc.sort_values(by="Perdida_COP", ascending=False).head(5).index
        auditoria_extremos = []
        for idx in peores_indices:
            sim_num = resultados_mc.loc[idx, "Simulacion"]
            secuencia = trayectorias_estados[idx]
            conteo_estados = {orden_estados[i]: secuencia.count(i) for i in range(3)}
            auditoria_extremos.append({
                "Simulación": sim_num,
                "Pérdida (COP)": f"${resultados_mc.loc[idx, 'Perdida_COP']:,.0f}",
                "Días en Adverso": conteo_estados["Adverso"],
                "Días en Neutral": conteo_estados["Neutral"],
                "Días en Favorable": conteo_estados["Favorable"],
                "Log-Retorno Portafolio (%)": f"{resultados_mc.loc[idx, 'Log_Ret_Portafolio_COP_pct']:.2f}%"
            })
        st.dataframe(pd.DataFrame(auditoria_extremos).set_index("Simulación"), width="stretch")

with tab5:
    st.subheader("Hallazgos, límites, decisión y bibliografía")

    p_brent = res_var.test_causality(caused="EC_ADR_ret", causing=["Brent_ret"], kind="f").pvalue
    p_cop = res_var.test_causality(caused="EC_ADR_ret", causing=["USD_COP_ret"], kind="f").pvalue
    chi2_est, p_indep, gl, _ = chi2_contingency(transiciones.to_numpy())

    st.markdown("**Hallazgos**")
    st.markdown(
        f"- **VAR({rezagos_var}):** el Brent y la tasa de cambio ayudan a predecir los rendimientos de EC "
        f"(causalidad de Granger, valor p = {p_brent:.4f} y {p_cop:.4f}). La correlación contemporánea entre EC y "
        f"Brent es {rendimientos['EC_ADR_ret'].corr(rendimientos['Brent_ret']):.3f}.\n"
        f"- **Markov:** los regímenes duran en promedio entre {min(duracion.values()):.2f} y {max(duracion.values()):.2f} días, "
        f"y la prueba chi cuadrado no rechaza la independencia entre estados consecutivos "
        f"(χ²({gl}) = {chi2_est:.2f}; valor p = {p_indep:.3f}).\n"
        f"- **Monte Carlo:** para {CAPITAL_INICIAL:,.0f} COP a {HORIZONTE} días, la probabilidad de pérdida es "
        f"{(perdidas > 0).mean() * 100:.2f} %, el VaR 95 % es {var_95:,.0f} COP y el VaR 99 % es {var_99:,.0f} COP."
    )
    st.success(
        f"**Decisión:** mantener una reserva de liquidez de {reserva_recomendada:,.0f} COP "
        f"(Expected Shortfall al 99 %, {reserva_recomendada / CAPITAL_INICIAL * 100:.1f} % del capital), "
        "tomada como mínimo y recalculada con datos recientes."
    )

    st.markdown("**Límites**")
    st.markdown(
        "- Los residuos del VAR no son normales y las funciones de impulso-respuesta dependen del orden de Cholesky "
        "(EC, Brent, USD/COP); se interpretan como dinámica predictiva, no como efectos causales estructurales.\n"
        "- La cadena de Markov casi no tiene memoria, así que la simulación no reproduce bien las rachas de días "
        "malos como las de marzo de 2020 y puede subestimar la cola extrema.\n"
        "- La muestra está dominada por el choque de la pandemia; el portafolio no incluye costos ni dividendos."
    )

    st.markdown("**Fuentes y bibliografía**")
    st.markdown(
        "- Yahoo Finance. (s. f.). Datos históricos de EC, BZ=F y COP=X. https://finance.yahoo.com/\n"
        "- Melo, L. F., & Hamann, F. A. (1998). *Inflación básica: Una estimación basada en modelos VAR estructurales* "
        "(Borradores de Economía No. 93). Banco de la República. https://doi.org/10.32468/be.93\n"
        "- Rosas Rosas, L. del C., & Segundo Rosas, J. I. (2019). Sobre cadenas de Markov y su potencial en las "
        "aplicaciones. *Epistemus, 13*(26), 7–12. https://doi.org/10.36790/epistemus.v13i26.90\n"
        "- Romero, P., & Rueda, R. (2013). Ars Conjectandi y el método Monte Carlo. *Revista Digital Universitaria, "
        "14*(11). https://www.revista.unam.mx/vol.14/num11/art44/\n"
        "- Hyndman, R. J., & Athanasopoulos, G. (2021). *Forecasting: Principles and practice* (3.ª ed.). OTexts. "
        "https://otexts.com/fpp3/\n"
        "- Seabold, S., & Perktold, J. (2010). Statsmodels: Econometric and statistical modeling with Python. "
        "*Proceedings of the 9th Python in Science Conference*, 92–96."
    )
