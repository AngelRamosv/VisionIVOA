# 👁️ IVOA (Inteligencia Operativa y Visión Artificial)
## Documentación Técnica y Arquitectónica del Sistema de Supervisión

---

## 1. Ficha Técnica y Resumen Ejecutivo

* **Nombre del Sistema:** IVOA - Inteligencia Operativa y Visión Artificial.
* **Propósito:** Auditoría y supervisión desatendida en tiempo real y por lotes (batch) de sucursales físicas de atención a clientes (sucursal izzi), evaluando el cumplimiento de SLAs, protocolos de seguridad, vestimenta y políticas de conducta del personal.
* **Paradigma de Diseño:** Arquitectura híbrida **Edge-to-Cloud**, desacoplada bajo el estándar **Open-Spec-Driven**.
* **Stack Tecnológico:**
  * **Lenguaje:** Python 3.11+.
  * **Edge Sensor (Procesamiento de Video):** Flask, OpenCV (captura RTSP y MP4), Ultralytics YOLOv8 (detección en cascada), Roboflow Supervision (ByteTrack, PolygonZone, LineZone, Annotators), UniFace con backend ONNX Runtime (SCRFD para detección facial y ArcFace MNet para biometría facial de 512 dimensiones), EasyOCR (validación de papel moneda).
  * **Cloud Backend (Orquestación & API):** FastAPI, Uvicorn, SQLAlchemy ORM, SQLite / PostgreSQL ready, Jinja2, Pydantic v2 (contratos de datos).
  * **Frontend & Visualización:** HTML5, CSS3 Glassmorphism (Dark Mode temático con acentos en tonos joya y degradados lineales), JavaScript vanilla asíncrono, Lucide Icons, Canvas API para dibujo interactivo de polígonos.
  * **Integraciones:** API independiente de mensajería para WhatsApp (OpenWA / Green-API compatible), exportación de métricas a Microsoft Excel (`.xlsx` vía OpenPyXL).

---

## 2. Arquitectura del Sistema y Flujo de Datos

### 2.1 Estructura del Repositorio (Open-Spec Driven)

```text
supervision/
├── config/
│   └── notify_config.json        # Ajustes de sucursal, flags de alertas, teléfonos y polígono activo
├── data/
│   └── ivoa.db                   # Base de datos SQLite (sesiones, eventos, hallazgos, usuarios)
├── docs/
│   ├── PRD_Videoanalitica_Sucursales_izzi.pdf   # Especificación de Requerimientos de Producto
│   ├── propuesta_comercial_vision_ia.html       # Propuesta ejecutiva y caso de negocio
│   └── DOCUMENTACION_SISTEMA_SUPERVISION_IVOA.md # Copia de respaldo de esta documentación
├── models/
│   ├── yolov8n.pt                # Modelo base COCO (personas, celulares, comida, periféricos)
│   ├── yolov8_ivoa_v5.pt         # Especialista en audífonos, celular, uniforme y comida
│   ├── yolov8_ivoa_v7.pt         # Especialista en monitores y páginas no autorizadas
│   ├── yolov8_ivoa_v8.pt         # Especialista en arreglo personal (maquillaje/cabello)
│   └── yolov8_ivoa_v9.pt         # Especialista en billetes, llaves y situación atípica
├── scripts/
│   ├── iniciar_todo.bat          # Lanzador concurrente de Edge Sensor y Cloud Backend
│   ├── iniciar_cloud.bat         # Lanzador del Cloud Backend (FastAPI :8001)
│   └── iniciar_sensor.bat        # Lanzador del Edge Sensor (Flask :5000)
├── spec/
│   └── openapi.yaml              # Especificación OpenAPI 3.1.0 de los contratos de la API
├── src/
│   ├── cloud_backend/            # Capa central de orquestación, persistencia y UI
│   │   ├── routers/
│   │   │   ├── events.py         # Endpoints de telemetría, logs, reportes y detonadores batch
│   │   │   └── views.py          # Rutas de renderizado web y autenticación
│   │   ├── static/               # CSS y JavaScript del Dashboard Central
│   │   ├── templates/            # Plantillas HTML (dashboard.html, login.html)
│   │   ├── database.py           # Conexión SQLAlchemy y modelos relacionales
│   │   ├── main.py               # Entrada principal FastAPI
│   │   ├── schemas.py            # Modelos Pydantic de entrada/salida
│   │   └── whatsapp_service.py   # Cliente HTTP puro para despacho a gateway de WhatsApp
│   └── edge_sensor/              # Nodo de captura de video e inferencia de IA local
│       ├── app_api.py            # Servidor Flask, hilos de streaming, grabación y análisis batch
│       ├── ivoa_engine.py        # Motor principal de inferencia en cascada y reglas de negocio
│       ├── biometric_engine.py   # Motor biométrico facial Edge con UniFace (SCRFD + ArcFace ONNX)
│       └── templates/            # Interfaz de depuración local del sensor
├── .gitignore                    # Reglas de exclusión de artefactos, pesos y temporales
├── docker-compose.yml            # Orquestación de contenedores
├── README.md                     # Documentación ejecutiva del repositorio
└── requirements.txt              # Dependencias fijadas del entorno Python
```

### 2.2 Diagrama de Flujo del Ecosistema

```text
       [ CÁMARA HIKVISION / RTSP ]  o  [ ARCHIVO MP4 DE BENCHMARK ]
                                   │
                                   ▼
                   ┌──────────────────────────────┐
                   │    IVOA Edge Sensor (:5000)   │
                   │  - Ingesta OpenCV            │
                   │  - Frame Skipping (1/60 seg) │
                   └──────────────┬───────────────┘
                                  │
                                  ▼
               ┌──────────────────────────────────────┐
               │         Pipeline IA en Cascada       │
               │  1. YOLOv8n (Personas, Celulares)    │
               │  2. YOLOv8 v5 (Uniforme, Audífonos)  │
               │  3. YOLOv8 v7 (Páginas Web Prohib.)  │
               │  4. YOLOv8 v8 (Arreglo Personal)     │
               │  5. YOLOv8 v9 (Billetes, Llaves)     │
               │  6. ByteTrack (Seguimiento temporal) │
               │  7. Validaciones geométricas cruzadas│
               └──────────────────┬───────────────────┘
                                  │
                 ┌────────────────┴────────────────┐
                 ▼                                 ▼
   [ Modo 1: Streaming En Vivo ]    [ Modo 2: Grabación Batch (10 min) ]
   - Flask MJPEG multipart          - Generación de incidencias.mp4
   - Alertas en DB silenciadas      - Análisis acelerado X2 (frame_skip=60)
                                    - Botón manual 'Enviar Alertas'
                                                   │
                                                   ▼
                                  ┌──────────────────────────────────┐
                                  │    IVOA Cloud Backend (:8001)    │
                                  │  - FastAPI / SQLAlchemy (ivoa.db)│
                                  │  - Pestañas: Live, KPIs, Ajustes │
                                  │  - Despacho consolidado WhatsApp │
                                  └──────────────────────────────────┘
```

---

## 3. Pipeline de Inteligencia Artificial (Inferencia en Cascada)

Para evitar la sobrecarga de un único modelo masivo y maximizar la precisión, IVOA implementa una **estrategia de inferencia en cascada modular**:

### 3.1 Modelos YOLO Utilizados
1. **`models/yolov8n.pt` (Universal Base Tracker):**
   * Detecta humanos (`class_id == 0`) con umbral estricto `conf >= 0.55` y filtro de altura mínima `bh >= 60px` para ignorar sombras, respaldos y sillas.
   * Detecta celulares estándar de frente (`class_id == 67` remapeado a `5`). Si el celular ocupa más del 12% del encuadre, activa sensibilidad alta (`conf >= 0.04`); si está lejos, requiere `conf >= 0.35`.
   * Identifica periféricos (`class_ids`: 62 pantalla, 63 laptop, 64 mouse, 66 teclado) utilizados para suprimir falsos positivos.
   * Detecta alimentos/bebidas COCO (`class_ids`: 39 botella, 40 copa, 41 taza, 46-55 comida) con umbral `0.18`.
2. **`models/yolov8_ivoa_v5.pt` (Especialista en Vestimenta y Accesorios):**
   * Detecta audífonos en oídos y uso de vestimenta/uniforme institucional corporativo.
3. **`models/yolov8_ivoa_v7.pt` (Especialista en Pantallas y Navegación No Autorizada):**
   * Identifica ventanas de navegación no permitidas (redes sociales, streaming, juegos) con `conf >= 0.04`.
   * **Filtro espacial anti-alucinación:** Se aplica la restricción geométrica `bw < width * 0.35` y `bh < height * 0.35`. Una pantalla o ventana de navegador dentro del módulo nunca supera el 35% de la imagen total, eliminando alucinaciones sobre espaldas o paredes.
4. **`models/yolov8_ivoa_v8.pt` (Especialista en Arreglo Personal):**
   * Detecta acciones de arreglo personal (uso de espejos, cepillos, maquillaje, retoques cosméticos) con `conf >= 0.25`.
   * Genera una caja delimitadora ajustada sobre el rostro/pecho evitando empalmarse con la etiqueta de tracking general.
5. **`models/yolov8_ivoa_v9.pt` (Especialista en Situación Atípica y Billetes):**
   * Detecta llaves expuestas (`conf >= 0.15`).
   * Detecta papel moneda / billetes (`conf >= 0.06`).
   * **Validación de Posesión Humana:** El billete solo se marca como válido si su caja delimitadora se superpone con la caja de una persona detectada (`is_held_by_person`). Si el billete se encuentra flotando en una pared o mueble sin interacción de manos, se descarta.
   * **Filtro Anti-Monitor:** Si el centro del supuesto billete coincide con las coordenadas de un monitor encendido, se descarta para evitar confusiones con gráficos o logotipos.

### 3.2 Algoritmo de Seguimiento Temporal (ByteTrack)
El rastreador multiobjeto `sv.ByteTrack` se encuentra parametrizado para la dinámica de una sucursal de alta afluencia:
* `track_activation_threshold = 0.55`: Requiere alta certeza para asignar un nuevo ID de cliente/asesor.
* `lost_track_buffer = 180`: Mantiene la memoria del ID durante 180 cuadros (~6 segundos a 30 FPS) en caso de que una persona sea momentáneamente ocluida por otra.
* `minimum_matching_threshold = 0.25`: Provee una vinculación 'pegajosa' (sticky tracking) que evita que el ID brinque erráticamente ante giros o movimientos rápidos.

### 3.3 Re-Identificación Biométrica Facial (UniFace + ONNX Runtime)
En `src/edge_sensor/biometric_engine.py`, el sistema implementa una arquitectura desacoplada de visión facial de alto rendimiento basada en el repositorio **UniFace**:

* **Detector de Rostros (SCRFD):**
  * Localiza rostros humanos con mínima carga computacional dentro del tercio superior (área de la cabeza) de la persona detectada por YOLO.
  * Extrae **5 landmarks anatómicos clave** (ojos, nariz y extremos de la boca) para realizar una transformación afín y normalizar la orientación del rostro antes de la inferencia.
* **Extractor de Características (ArcFace MobileNet):**
  * Ejecutado sobre el runtime optimizado de **ONNX Runtime (C++)**, generando un vector de embedding facial denso y normalizado (\(L_2\)) de **512 dimensiones**.
  * Reemplaza por completo el modelo de aproximación textil por PyTorch, logrando tiempos de inferencia de apenas **5 a 10 ms por rostro en CPU**, sin fuga de memoria ni sobrecalentamiento del sensor.
* **Mecánica de Gating y Aceleración en RAM:**
  1. **Filtro de Resolución:** Solo procesa cajas delimitadoras de personas con altura mayor a 60 píxeles para evitar inferencias en siluetas lejanas o borrosas.
  2. **Inferencia Espaciada:** El sensor solo evalúa la biometría cada 15 cuadros (`frame_count % 15 == 0`) y exclusivamente en sujetos con ID de tracking no reconocido.
  3. **Caché en Memoria RAM (`_db_cache`):** Los vectores faciales almacenados en la base de datos relacional (`data/ivoa.db`) se precargan en memoria como arrays NumPy. La comparación de similitud de coseno contra el catálogo de identidades toma fracciones de milisegundo mediante operaciones vectorizadas de álgebra lineal:
     $$\text{Similitud} = \frac{\mathbf{u} \cdot \mathbf{v}}{\|\mathbf{u}\|_2 \|\mathbf{v}\|_2}$$
  4. **Persistencia y Congelación:** En cuanto un sujeto supera el umbral de coincidencia facial (`threshold_cara = 0.50`), su identidad se vincula a su `tracker_id` en el diccionario local `self.known_identities`. Una vez identificado, la inferencia biométrica se apaga para ese sujeto mientras permanezca en escena, preservando los FPS originales de la cámara.

---

## 4. Reglas de Negocio, Clasificación de Incidencias y Zonas

El motor `IVOAVisionEngine` procesa cada cuadro evaluando las siguientes reglas de negocio:

| Event Type / Incidencia | Categoría en Dashboard | Regla de Disparo / Condición | Política de Disparo |
|---|---|---|---|
| `late_opening` | Apertura tardía | Apertura después de las 09:00 AM o retraso de inicio de atención | Única vez por turno |
| `incidencia_uniforme` | Código de vestimenta | Asesor sin playera/camisa institucional o con prenda indebida | Única vez por turno |
| `incidencia_celular` | Uso de celulares | Celular en mano en zona de atención > 1.0 segundo continuo | Múltiples alertas permitidas |
| `incidencia_situacion_atipica` | Situación atípica | Llaves o desorden crítico en el mostrador | Única vez por turno |
| `incidencia_dinero` | Billetes dinero | Billetes fuera de caja registradora en zona de mostrador | Múltiples alertas permitidas |
| `incidencia_alimentos` | Consumo de alimentos | Botellas, vasos, comida sobre el mostrador de atención | Múltiples alertas permitidas |
| `incidencia_audifonos` | Uso de audífonos | Audífonos en oído durante horario laboral en ventanilla | Múltiples alertas permitidas |
| `incidencia_paginas` | Páginas no autorizadas | Pantalla con navegación recreativa o no autorizada | Múltiples alertas permitidas |
| `incidencia_arreglo_personal` | Arreglo personal | Maquillaje, peinado o uso de espejo en atención | Múltiples alertas permitidas |
| `customer_abandonment` | Abandono de cliente | Cliente sale de la fila sin haber pasado al módulo | Tiempo en fila > umbral |

### Zonas Espaciales y Geometría Interactiva
* **`fila_zone`:** Polígono inferior donde se calcula el aforo y tiempos de espera en cola.
* **`modulo_zone`:** Polígono superior donde se audita el comportamiento del asesor y la atención directa.
* **`custom_polygon`:** Polígono configurable dinámicamente desde el Dashboard con el mouse (vía Canvas API). Los puntos se normalizan (0.0 a 1.0) y se almacenan en `config/notify_config.json`, restringiendo la supervisión a la zona dibujada por el usuario.
* **Debouncing y Cooldown:**
  * `DEBOUNCE_SECONDS = 1.0`: La infracción debe sostenerse 1 segundo ininterrumpido para ser tomada en cuenta.
  * `ALERT_COOLDOWN = 8.0`: Ventana de silencio de 8 segundos entre alertas idénticas para evitar inundación de eventos.

---

## 5. Modos de Operación: Monitoreo en Vivo vs. Modo Lote (Batch Offline)

### 5.1 Modo Monitoreo en Vivo (Live Stream)
* Se sirve en la ruta `/video_feed` de Flask a través de un stream MJPEG `multipart/x-mixed-replace`.
* **Regla de silencio en WhatsApp:** Durante el monitoreo normal en vivo, el sistema **no** envía mensajes de WhatsApp para no saturar al personal mientras atiende a clientes. Todos los eventos se acumulan en la base de datos `data/ivoa.db`.

### 5.2 Modo Batch (Grabación de 10 minutos y Análisis Offline Acelerado)
* **Captura:** Al detonar `/api/start_batch`, el sistema graba en `C:/Users/Angel/Desktop/prueba/incidencias.mp4` exactamente 10 minutos de video (600 segundos) a 10 FPS (6,000 cuadros en total) a una resolución de 1280x720.
* **Motor Offline Acelerado (X2 / Modo Turbo):** Al terminar la grabación, se lanza el hilo en segundo plano `offline_analyze_thread`.
  * Parámetro: `offline_engine = IVOAVisionEngine(width=1280, height=720, frame_skip=60)`.
  * Al saltar 60 cuadros (analizando 1 cuadro cada 6 segundos de tiempo real), el sistema evalúa únicamente ~100 cuadros clave en lugar de los 6,000 cuadros brutos.
  * **Resultado:** El tiempo de análisis de un bloque de 10 minutos se redujo de 670 segundos (~11 minutos) a **menos de 60 segundos** en CPU estándar.
* **Despacho Consolidado:** Al concluir el análisis, el usuario hace clic en el botón *'Enviar Alertas'* (`/v1/events/trigger_batch`). El sistema recopila las incidencias registradas en los últimos 15 minutos, arma un resumen ejecutivo limpio con emojis y viñetas horarias, y lo envía en un solo mensaje de WhatsApp. Posteriormente, elimina el archivo temporal `incidencias.mp4` para liberar almacenamiento en disco.

---

## 6. Contratos de API, Persistencia y Notificaciones

### 6.1 Base de Datos Relacional (`data/ivoa.db`)
El sistema utiliza SQLAlchemy con 4 tablas principales:
1. **`usuarios`:** Control de acceso al Dashboard Cloud (`admin` / `admin123`).
2. **`sesiones_personas`:** Registra cada `tracker_id`, timestamps de entrada/salida a la fila, entrada/salida al módulo, tiempo total de atención y embeddings de ropa/rostro.
3. **`eventos`:** Bitácora histórica con `event_type`, `timestamp`, `details` (JSON) y `severity` (`info`, `warning`, `critical`).
4. **`hallazgos`:** Tickets operacionales generados a partir de eventos, con seguimiento de estado (`DETECTADO`, `CERRADO`) y fecha de expiración para cálculo de SLAs.

### 6.2 Servicio Autónomo de WhatsApp (`src/cloud_backend/whatsapp_service.py`)
* Diseñado como un cliente HTTP autónomo sin librerías pesadas de terceros (utiliza la librería estándar `urllib.request`).
* Normaliza cualquier número mexicano (`+52155...`, `55...`) al formato JID `@c.us` o identificadores de grupo `@g.us`.
* Se conecta a la pasarela OpenWA en `http://192.168.49.93:2785` con llave maestra autenticada (`mAGD3357$`).

---

## 7. Guía de Ejecución y Despliegue Rápido

1. **Requisitos de Entorno e Instalación:**
   * Windows 10/11 x64.
   * Python 3.11 instalado en el entorno `.venv`.
   * **Instalación de paquetes y motor ONNX:**
     ```powershell
     .\.venv\Scripts\activate
     pip install -r requirements.txt
     pip install -e C:\Users\Angel\uniface
     ```
2. **Iniciar la plataforma completa:**
   * Ejecutar el script `scripts/iniciar_todo.bat` con doble clic.
   * Esto abrirá automáticamente dos ventanas de consola:
     * **Consola 1 (Cloud Backend):** `http://localhost:8001` (Credenciales: `admin` / `admin123`).
     * **Consola 2 (Edge Sensor):** `http://localhost:5000`.
3. **Acceso al Dashboard:**
   * Abrir un navegador web en: `http://localhost:8001/dashboard`.

---

## 8. Sección de Ingeniería: Mejoras Potenciales, Modos de Falla y Guía de Soluciones

Esta sección identifica los puntos de tensión técnica, posibles fallos en entornos de producción y su resolución exacta.

---

### 8.1 Falsos Positivos y Ruido Visual

#### A. Alucinación de Páginas No Autorizadas sobre Espaldas de Clientes
* **Qué puede fallar:** En sucursales con ropa estampada, cuadros o texturas geométricas en camisas oscuras, el modelo v7 puede confundir la espalda de una persona con la pantalla de una computadora.
* **Mecanismo actual implementado:** Filtro de escala estricto (`bw < self.width * 0.35` y `bh < self.height * 0.35`). Las espaldas superan comúnmente este tamaño y son descartadas.
* **Mejora recomendada:** Activar una máscara de intersección con la clase 62/63 (Monitor/Laptop de COCO). Una página solo debe aceptarse si se encuentra matemáticamente contenida dentro del área de un monitor detectado por `yolov8n.pt`.

#### B. Detección de Billetes en Superficies Inertes
* **Qué puede fallar:** Carpetas beige, sobres amarillos o volantes comerciales en el mostrador pueden confundirse con billetes de denominaciones mexicanas.
* **Mecanismo actual implementado:** Filtro de posesión humana (`is_held_by_person`). El billete se descarta si no está en un radio de 80 píxeles respecto a una persona.
* **Mejora recomendada:** Habilitar el módulo EasyOCR integrado (`self._ocr_reader`) únicamente cuando el billete permanezca estático más de 2 segundos, buscando patrones numéricos ('20', '50', '100', '200', '500') para confirmar la transacción bancaria.

---

### 8.2 Estabilidad del Enlace de Video (Cámara IP Hikvision / RTSP)

#### A. Congelamiento o Caída del Stream RTSP por Pérdida de Paquetes
* **Qué puede fallar:** Las cámaras IP en redes locales corporativas suelen sufrir caídas momentáneas por saturación de ancho de banda o switches PoE, provocando que `cv2.VideoCapture` se bloquee indefinidamente.
* **Mecanismo actual implementado:** Variables de entorno aplicadas en `app_api.py`:
  ```python
  os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
  ```
  Esto obliga a la cámara a transmitir por TCP en lugar de UDP, evitando que se pierdan paquetes o se corrompan frames.
* **Solución si el sensor se queda congelado:**
  1. Verificar ping continuo a la cámara: `ping 192.168.51.125 -t`.
  2. Implementar un hilo 'Watchdog' en `CameraReader` que verifique si han pasado más de 5 segundos sin recibir un frame nuevo; de ser así, invocar `cap.release()` y reinicializar la conexión automáticamente.

---

### 8.3 Concurrencia y Bloqueos en Base de Datos (SQLite Locking)

#### A. Error `sqlite3.OperationalError: database is locked`
* **Qué puede fallar:** Dado que tanto el sensor (`app_api.py`) como el backend (`main.py`) escriben de forma asíncrona en `data/ivoa.db`, en picos altos de concurrencia SQLite puede rechazar escrituras por bloqueo de archivo.
* **Mecanismo actual implementado:** `connect_args={"check_same_thread": False}` y sesiones transaccionales con cierre forzado en bloques `finally`.
* **Solución definitiva para escalabilidad empresarial:**
  * Migrar a **PostgreSQL**. El código ya utiliza SQLAlchemy; basta con definir la variable de entorno:
    ```bash
    set DATABASE_URL=postgresql://usuario:password@localhost:5432/ivoa_db
    ```
  * O bien, activar el modo WAL (Write-Ahead Logging) en SQLite ejecutando una sola vez:
    ```sql
    PRAGMA journal_mode=WAL;
    ```

---

### 8.4 Saturación de Hardware y Rendimiento de Inferencia

#### A. Caída de FPS en Equipos sin Tarjeta Gráfica Dedicada (CPU Only)
* **Qué puede fallar:** Ejecutar 5 modelos YOLO simultáneamente en una computadora de oficina sin GPU dedicada puede elevar el uso de CPU al 100%, ralentizando el video.
* **Soluciones y Optimizaciones Disponibles:**
  1. **Ajuste de Salto de Cuadros en Tiempo Real:** En `ivoa_engine.py`, cambiar `frame_skip=5` a `frame_skip=10` o `frame_skip=15` para el stream en vivo.
  2. **Exportación a OpenVINO o ONNX:** Convertir los pesos `.pt` de Ultralytics al formato acelerado de Intel:
     ```python
     from ultralytics import YOLO
     YOLO("models/yolov8_ivoa_v5.pt").export(format="openvino")
     ```
     Esto reduce el consumo de CPU hasta en un 60% en procesadores Intel Core i5/i7.
  3. **Exportación a TensorRT (si se cuenta con GPU NVIDIA):**
     ```python
     YOLO("models/yolov8n.pt").export(format="engine")
     ```

---

### 8.5 Entrega de Alertas por WhatsApp

#### A. Mensajes no Entregados o Timeout en Servidor OpenWA
* **Qué puede fallar:** Si la IP `192.168.49.93` cambia o el teléfono vinculado se queda sin batería/conexión, la API de OpenWA puede responder con código 500 o no contestar.
* **Mecanismo de mitigación y diagnóstico:**
  1. Ejecutar prueba manual en consola:
     ```bash
     python -c "from src.cloud_backend.whatsapp_service import send_whatsapp_message; send_whatsapp_message('Test manual')"
     ```
  2. Revisar si la sesión sigue activa accediendo en el navegador a `http://192.168.49.93:2785/api/sessions`.
  3. Si la sesión está caída, escanear nuevamente el código QR desde el panel de administración de OpenWA.

---

### 8.6 Matriz de Resumen de Problemas Frecuentes

| Síntoma Observado | Causa Raíz Probable | Solución Rápida |
|---|---|---|
| La pantalla del video en vivo se queda en negro o dice 'Cámara desconectada'. | Cable de red desconectado, IP de cámara cambió o RTSP saturado. | Verificar alimentación PoE de la cámara Hikvision y revisar URL en `app_api.py`. |
| El análisis por lote tarda mucho tiempo. | `frame_skip` fue modificado a un valor bajo. | Comprobar que en `app_api.py` línea 344 `frame_skip` sea igual a `60`. |
| El reporte de WhatsApp no llega tras presionar 'Enviar Alertas'. | No hubo incidencias en la ventana de 15 minutos o teléfono mal configurado. | Ir a la pestaña *Ajustes de Incidencias* en el Dashboard, validar el número de teléfono con lada internacional (+521...) y guardar. |
| El diseño de la tabla se ve desalineado. | Caché agresiva del navegador web. | Presionar `Ctrl + F5` en el navegador para forzar la recarga de estilos CSS. |

---

*Documentación técnica consolidada para el proyecto IVOA - Supervisión Inteligente.*
