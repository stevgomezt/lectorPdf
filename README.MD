
# /proyecto_lector_pdf

# C:\Users\sgomezt\Documents\Proyectos\proyecto_lector_pdf

virtualenv -p python3 venv_lector_pdf
.\venv_lector_pdf\Scripts\activate
deactivate
pip install -r requirements.txt
python app.py
http://127.0.0.1:5000/

proyecto_lector_pdf/
│
├── venv_lector_pdf/        # Entorno virtual (puede variar en sistemas Windows)
├── src/                    # Carpeta para el código fuente
│   ├── __init__.py         # Archivo que hace que Python trate el directorio como un paquete
│   ├── app.py              # Archivo principal de Flask
│   ├── pdf_processor.py    # Archivo para las funciones de procesamiento de PDF
│   └── database.py         # Archivo para las funciones de interacción con la base de datos
│
├── static/                 # Carpeta para archivos estáticos (CSS, JS, imágenes, etc.)
│   └── css/                # Carpeta para archivos CSS
│       └── style.css       # Archivo CSS (puedes crearlo más tarde)
│   └── img/                # Carpeta para imágenes (puedes crearla más tarde)
│
├── templates/              # Carpeta para plantillas HTML
│   ├── index.html          # Página principal
│   └── resultados.html     # Página de resultados
│
├── requirements.txt        # Archivo para las dependencias del proyecto


Sección 1:
Parte 1: Carga del PDF.
Parte 2: Verificación del contenido para reconocer el contenido de la página de Gráfico Perfil Conductual.

Sección 2:
Parte 1: Lectura para PDF y reconocer información por medio del lector OCR, con las librerias de OCRmyPDF y Tesseract. (A continuación se encuentra la explicación de las librerias)
Parte2: Realiza la tabulación de la información de cada segmento de la página del PDA en estructura de diccionario.

Sección 3:
Parte 1: Almacenamiento del Diccionario en json, csv u otros formatos.
