@st.cache_data
def cargar_datos():
    ruta = "base_precios_depurada_2019_2021_LosEndeudados_v2.csv"
    df = pd.read_csv(ruta, sep=None, engine="python")
    df.columns = [str(col).strip() for col in df.columns]

    # Buscar columna de fecha (Fecha, fecha, Date, etc.)
    col_fecha = next((c for c in df.columns if c.lower() in ["fecha", "date"]), df.columns[0])
    df[col_fecha] = pd.to_datetime(df[col_fecha], format="mixed", dayfirst=True, errors="coerce")
    df = df.dropna(subset=[col_fecha]).sort_values(col_fecha).set_index(col_fecha)

    cols_precios = ["EC_ADR_USD_Cierre", "Brent_USD_Cierre", "USD_COP_Cierre"]
    faltantes = [c for c in cols_precios if c not in df.columns]
    if faltantes:
        return pd.DataFrame(), pd.DataFrame(), list(df.columns), df.head(3)

    precios = df[cols_precios].copy()
    for c in cols_precios:
        if precios[c].dtype == "object":
            s = precios[c].astype(str).str.strip().str.replace(" ", "", regex=False)
            mask_ambos = s.str.contains(r"\.", regex=True) & s.str.contains(",", regex=True)
            s.loc[mask_ambos] = (
                s.loc[mask_ambos]
                .str.replace(".", "", regex=False)
                .str.replace(",", ".", regex=False)
            )
            mask_coma = ~mask_ambos & s.str.contains(",", regex=True)
            s.loc[mask_coma] = s.loc[mask_coma].str.replace(",", ".", regex=False)
            precios[c] = pd.to_numeric(s, errors="coerce")
        else:
            precios[c] = pd.to_numeric(precios[c], errors="coerce")

    precios = precios.dropna()
    rendimientos = 100 * np.log(precios / precios.shift(1)).dropna()
    rendimientos.columns = ["EC_ADR_ret", "Brent_ret", "USD_COP_ret"]
    return precios, rendimientos, list(df.columns), precios.head(3)


precios, rendimientos, cols_detectadas, muestra_df = cargar_datos()

if rendimientos.empty:
    st.error("No se pudieron calcular rendimientos válidos. Revisa los nombres de columnas o el formato numérico del CSV.")
    st.write("Columnas detectadas en el CSV:", cols_detectadas)
    st.dataframe(muestra_df)
    st.stop()
