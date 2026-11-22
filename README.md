# Monitor de Riesgo Bursátil: Choques Externos sobre Ecopetrol ADR

Proyecto integrador del curso **Análisis e Interpretación de Datos a través de Python** (tercer corte), Universidad Católica de Colombia. Profesor: Michael Smith Ortegón Salazar.

**Equipo «Los Endeudados»:** Harold David Colmenares Rodríguez, Alejandro González Bedoya y Santiago Rivas Raquejo.

**Aplicación publicada:** https://ccwcakraf57zwzp4acheza.streamlit.app/

## Pregunta central

¿Cómo se transmiten los choques del Brent y de la tasa de cambio USD/COP sobre los rendimientos de Ecopetrol ADR entre 2019 y 2021, con qué probabilidad transita el activo entre regímenes adverso, neutral y favorable, y qué pérdida extrema en pesos enfrentaría una inversión a 21 días hábiles?

- **Usuario:** analista de riesgo de un portafolio en pesos con posición en Ecopetrol ADR.
- **Decisión:** asignar una reserva de liquidez igual al Expected Shortfall al 99 % de la pérdida simulada.

## Datos

| Variable | Unidad | Fuente (ticker) | Función |
|---|---|---|---|
| Ecopetrol ADR | USD por acción | Yahoo Finance (`EC`) | Variable objetivo |
| Petróleo Brent | USD por barril | Yahoo Finance (`BZ=F`) | Choque externo |
| Tasa de cambio | COP por USD | Yahoo Finance (`COP=X`) | Choque externo y conversión a pesos |

- Frecuencia diaria, del 2 de enero de 2019 al 30 de diciembre de 2021 (días hábiles comunes a las tres series).
- 755 precios y 754 rendimientos logarítmicos por serie, sin datos faltantes.
- El archivo `base_precios_depurada_2019_2021_LosEndeudados_v2.csv` se generó con el cuaderno `PROYECTO.ipynb`, que descarga las series con `yfinance`, alinea las fechas comunes y redondea los precios a cuatro decimales. El CSV usa `;` como separador y formato numérico colombiano; `app.py` lo interpreta con la función `limpiar_numero_serie`.

## Métodos

1. **Datos y estacionariedad:** rendimientos logarítmicos `r = 100 × ln(P_t / P_{t-1})`, estadísticas descriptivas, correlaciones y prueba ADF.
2. **Modelo VAR:** VAR(1), VAR(3) o VAR(6) (por defecto 6, elegido por AIC); diagnósticos de estabilidad, autocorrelación y normalidad; funciones de impulso-respuesta ortogonalizadas a 10 días y pronóstico a 10 días.
3. **Cadenas de Markov:** tres estados (adverso, neutral y favorable) según los terciles del rendimiento de EC; matriz de transición, duración esperada y distribución estacionaria.
4. **Simulación Monte Carlo:** remuestreo histórico condicionado al estado de Markov (semilla 42); VaR y Expected Shortfall al 95 % y 99 % de una inversión en pesos.
5. **Conclusiones:** hallazgos, límites, decisión y bibliografía, calculados con los parámetros elegidos en la barra lateral.

## Estructura del repositorio

```
├── app.py                                               # Aplicación de Streamlit
├── PROYECTO.ipynb                                       # Descarga, limpieza y análisis exploratorio (Google Colab)
├── base_precios_depurada_2019_2021_LosEndeudados_v2.csv # Base de precios depurada
├── requirements.txt                                     # Librerías necesarias
└── README.md
```

## Cómo ejecutar la aplicación localmente

Requiere Python 3.10 o superior.

```bash
git clone https://github.com/haroldcolmenaresrodriguez-cyber/Proyecto-Integrador-.git
cd Proyecto-Integrador-
pip install -r requirements.txt
streamlit run app.py
```

La aplicación se abre en el navegador en `http://localhost:8501`. El CSV debe estar en la misma carpeta que `app.py`.

## Cómo publicarla en Streamlit Community Cloud

1. Subir `app.py`, `requirements.txt` y el CSV a la rama `main` del repositorio en GitHub.
2. Entrar a https://share.streamlit.io con la cuenta de GitHub y elegir **Create app**.
3. Seleccionar el repositorio, la rama `main` y el archivo principal `app.py`, y pulsar **Deploy**.
4. Cada vez que se sube un cambio a `main`, la aplicación se actualiza sola.

## Bibliografía

- Hyndman, R. J., & Athanasopoulos, G. (2021). *Forecasting: Principles and practice* (3.ª ed.). OTexts. https://otexts.com/fpp3/
- Melo, L. F., & Hamann, F. A. (1998). *Inflación básica: Una estimación basada en modelos VAR estructurales* (Borradores de Economía No. 93). Banco de la República. https://doi.org/10.32468/be.93
- Romero, P., & Rueda, R. (2013). Ars Conjectandi y el método Monte Carlo. *Revista Digital Universitaria, 14*(11). https://www.revista.unam.mx/vol.14/num11/art44/
- Rosas Rosas, L. del C., & Segundo Rosas, J. I. (2019). Sobre cadenas de Markov y su potencial en las aplicaciones. *Epistemus, 13*(26), 7–12. https://doi.org/10.36790/epistemus.v13i26.90
- Seabold, S., & Perktold, J. (2010). Statsmodels: Econometric and statistical modeling with Python. En *Proceedings of the 9th Python in Science Conference* (pp. 92–96).
