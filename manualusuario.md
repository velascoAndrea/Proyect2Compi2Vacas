# Manual del usuario
## Coronavirus Data Analysis With Machine Learning

Con la llegada del virus COVID-19 y variaciones se ha requerido requerido una aplicación web que permita analizar el comportamiento del virus en el mundo, de esa forma se puede estudiar y tomar medidas preventivas para evitar su propagación. Es aquí donde entra la aplicación Coronavirus Data Analysis With Machine Learning que permite hacer un análisis minucioso del comportamiento del virus.

### Áreas
La aplicación cuenta con 3 áreas:

  * **Inicio**: Se selecciona un archivo de entrada.
  * **Parametrización**: Se seleccionan el tipo de predicción y los campos necesarios para realizar el análisis.
  * **Reportes**: Es el área donde muestra las gráficas y las conclusiones del análisis realizado.

### Flujo de la aplicación

  * Ingresamos a la aplicación web: [Coronavirus Data Analysis With Machine Learning](https://proyecto2compivacas.herokuapp.com/)

> En la pantalla de inicio podemos seleccionar el archivo de entrada con el que se realizará la predicción respectiva. La aplicación soporta tres tipos archivos de entrada:

  > * **CSV** : Archivo de texto delimitado por comas.
  >* **Excel** : Archivos de Microsoft Excel (.xlsx, xls).
  >* **JSON** : Achivo json.

![selecionar archivo](./images/inicio2.png)

> Una vez seleccionado el archivo de entrada debemos hacer clic en el botón **Subir**. Al realizar esto se mostrará el archivo cargado.

![archivos cargados](./images/inicio3.png)

  * Seleccionamos el menú [Parametros](https://proyecto2compivacas.herokuapp.com/about)

>En esta pantalla podemos seleccionar la prediccion deseada.

![seleccion de prediccion](./images/param1.png)

>Una ves seleccionada la predicción, podemos mapear las distintas columnas necesarias para el analisis. Una vez hecho esto debemos hacer clic en el botón **Confirmar Parametrización**.

![seleccion de columnas](./images/param2.png)

>Por ultimo debemos seleccionar el continente, región, país, departamento, el cual se debe aplicar la predicción seleccionada. Una vez hecho esto debemos hacer clic en el botón **Analizar Parametros**.

![seleccion de columnas](./images/param3.png)

  * Para ver los resultados debemos haci clic en el menú [Reportes](https://proyecto2compivacas.herokuapp.com/reportes)