@st.cache_data
def cargar_datos():
    ruta = "base_precios_depurada_2019_2021_LosEndeudados_v2.csv"
    df = pd.read_csv(ruta, sep=None, engine="python")
    df["Fecha"] = pd.to_datetime(df["Fecha"], format="mixed", dayfirst=True, errors="coerce")
    df = df.dropna(subset=["Fecha"]).sort_values("Fecha").set_index("Fecha")

    cols_precios = ["EC_ADR_USD_Cierre", "Brent_USD_Cierre", "USD_COP_Cierre"]
    precios = df[cols_precios].copy()

    for c in cols_precios:
        if precios[c].dtype == "object":
            s = precios[c].astype(str).str.strip().str.replace(" ", "", regex=False)
            # Si tiene coma y punto, asumimos punto como miles y coma como decimal
            mask_ambos = s.str.contains(r"\.", regex=True) & s.str.contains(",", regex=True)
            s.loc[mask_ambos] = (
                s.loc[mask_ambos]
                .str.replace(".", "", regex=False)
                .str.replace(",", ".", regex=False)
            )
            # Si solo tiene coma, la convertimos a punto decimal
            mask_coma = ~mask_ambos & s.str.contains(",", regex=True)
            s.loc[mask_coma] = s.loc[mask_coma].str.replace(",", ".", regex=False)
            precios[c] = pd.to_numeric(s, errors="coerce")
        else:
            precios[c] = pd.to_numeric(precios[c], errors="coerce")

    precios = precios.dropna()
    rendimientos = 100 * np.log(precios / precios.shift(1)).dropna()
    rendimientos.columns = ["EC_ADR_ret", "Brent_ret", "USD_COP_ret"]
    return precios, rendimientos
