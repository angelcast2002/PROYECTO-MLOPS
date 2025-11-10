# Informe Final del Proyecto

Título: Clasificación de Documentos en Español con Enfoque MLOps  
Versión del sistema: 0.1.6  
Fecha: 10/11/2025  
Autores: Angel Castellanos, Alejandro Azurdia, Diego Morales  
Repositorio: https://github.com/angelcast2002/PROYECTO-MLOPS  
Paquete PyPI: https://pypi.org/project/proyecto-mlops/  
Docker Hub: https://hub.docker.com/repository/docker/angelcast2025/proyecto-mlops/general

---

## 1. Introducción
El presente informe documenta, de forma narrativa y orientada a decisión, el desarrollo e industrialización de un sistema de clasificación automática de documentos en español. El proyecto siguió la metodología CRISP‑DM y se apoyó en prácticas de Ingeniería de Machine Learning (MLOps) para garantizar reproducibilidad, trazabilidad y despliegue confiable. La solución incluye un paquete Python publicable en PyPI, una imagen de contenedor Docker lista para su ejecución en servidores estándar y una automatización de entrega continua con GitHub Actions. Además, se estableció un mecanismo de versionado de datos y modelos que permite auditar cambios y mantener gobernanza sobre el ciclo de vida.

La motivación de negocio es reducir de manera sustancial el tiempo y el costo del procesamiento manual de documentos. El sistema automatiza la clasificación y facilita la escalabilidad operativa, la uniformidad de criterios y la disponibilidad 24/7. A partir de un conjunto de datos de referencia, se diseñó una cadena de valor donde los pasos de preparación de datos, entrenamiento, evaluación y despliegue pueden ejecutarse de manera repetible y medible.

## 2. Objetivos y Alcance
El objetivo general fue construir y operacionalizar un clasificador de documentos en español, cubriendo extremo a extremo las fases necesarias para pasar de un experimento a un entregable de producción. El alcance comprendió la preparación de datos, el modelado, la evaluación, el empaquetado del código, la configuración de pipelines de CI/CD, la construcción de la imagen Docker y la elaboración de una guía de despliegue en DigitalOcean. Se definieron criterios de éxito pragmáticos: alcanzar F1‑macro ≥ 0.75, publicar el paquete en PyPI, disponibilizar la imagen en Docker Hub y ejecutar los pipelines automatizados sin intervención manual.

## 3. Metodología (CRISP‑DM)
El trabajo siguió la estructura CRISP‑DM y cada etapa se materializó en módulos de código claramente separados. En Business Understanding se clarificó el valor esperado por la organización: acelerar el procesamiento y homogeneizar criterios, con niveles de servicio que contemplan latencias operativas adecuadas. En Data Understanding se realizó la carga y exploración del dataset, se definió el esquema de datos y se implementaron validaciones básicas para detectar inconsistencias de forma temprana. En Data Preparation se normalizó el texto (minúsculas, manejo de acentos, Unicode), se tokenizó con expresiones regulares y se ejecutó una limpieza prudente que incluye la eliminación de stopwords mediante NLTK y el uso de stemming en español (Snowball) para mejorar la generalización.

En la fase de Modeling se optó por un enfoque robusto y eficiente: representaciones TF‑IDF con n‑gramas de tamaño uno y dos, combinadas con un clasificador LinearSVC. Este diseño ofrece una relación favorable entre precisión, interpretabilidad y costo computacional. Se evaluó con validación simple (holdout) y validación cruzada de cinco particiones, y se añadió un barrido de hiperparámetros acotado para afinar los principales grados de libertad del modelo. En Evaluation se consolidaron métricas de exactitud y F1 (macro y ponderada), se midieron latencias de inferencia en percentiles operativos (P50, P95, P99) y se introdujeron controles de equidad revisando el desempeño mínimo por clase. Finalmente, en Deployment se serializaron artefactos con joblib, se registraron versiones y se empaquetó el proyecto para su publicación en PyPI y su ejecución dentro de un contenedor Docker.

## 4. Arquitectura de la Solución
La solución está organizada como un paquete Python en el directorio `proyecto_mlops`, con submódulos que corresponden a cada fase CRISP‑DM. Los datos y modelos se versionan bajo `data/` y `models/`, complementados por metadatos que registran fechas, parámetros y rutas. La automatización se apoya en GitHub Actions: existe una canalización de publicación a PyPI que se activa al crear un tag de versión, un flujo de construcción y publicación de la imagen Docker hacia Docker Hub y un pipeline de registro de modelos que ejecuta el procesamiento y sube artefactos con política de retención. La imagen Docker se basa en Python 3.11‑slim con dependencias declaradas en `pyproject.toml`, lo que simplifica su ejecución en múltiples entornos.

## 5. Implementación y Desarrollo
Para la preparación de datos se implementó un flujo resistente a fallos de recursos externos: si las listas de stopwords de NLTK no están presentes, el sistema intenta descargarlas y, en casos límite, continúa con una degradación controlada que mantiene la ejecución estable. El entrenamiento del modelo se realiza con scikit‑learn y los artefactos resultantes se guardan en disco de manera estructurada. La interacción con el sistema puede hacerse a través de un CLI (`proyecto-mlops`) que permite ejecutar tanto fases individuales como un pipeline completo. El proceso de publicación en PyPI incluye una verificación de instalación posterior a la subida para asegurar que el paquete es consumible. La construcción de la imagen Docker se define desde `infra/Dockerfile` y su publicación se realiza en el repositorio `angelcast2025/proyecto-mlops`.

## 6. Resultados y Evaluación
Las evaluaciones internas muestran niveles de desempeño acordes a los objetivos, con valores de exactitud alrededor de 0.82 y F1‑macro alrededor de 0.78 en el conjunto de prueba de referencia. En términos operativos, las mediciones de latencia se sitúan dentro de umbrales adecuados para un servicio batch o semi‑interactivo, con especial atención al percentil 95 como indicador de experiencia del usuario. Para mitigar sesgos o degradaciones puntuales, se vigila que el F1 por clase no descienda por debajo de una cota razonable. Toda esta evidencia se materializa en reportes exportados a archivos JSON y CSV, así como en documentos de apoyo.

## 7. Discusión y Análisis
El esquema TF‑IDF con LinearSVC representa un compromiso efectivo entre rendimiento y coste. Frente a alternativas neurales más pesadas, esta combinación tiene un ciclo de entrenamiento y una huella de inferencia más reducida, lo que facilita despliegues en servidores de capacidades acotadas. En escenarios con mayor variabilidad semántica o requisitos de comprensión contextual profunda, se considera apropiada la transición hacia representaciones embebidas (Word2Vec o fastText) o incluso modelos basados en Transformers, siempre que el presupuesto y los objetivos de negocio lo justifiquen. Esta evolución puede abordarse gradualmente, conservando el andamiaje MLOps ya establecido.

## 8. Despliegue en Producción (DigitalOcean)
El sistema está preparado para su ejecución en un Droplet estándar de DigitalOcean. La guía `docs_project/DEPLOY_DIGITAL_OCEAN.md` describe, paso a paso, cómo instalar Docker, iniciar sesión en Docker Hub, descargar la imagen y ejecutar el contenedor con los volúmenes y puertos necesarios. También se incluye la configuración de un servicio `systemd` para asegurar el reinicio automático y la continuidad operativa ante reinicios del servidor. La imagen publicada `angelcast2025/proyecto-mlops:0.1.6` puede sustituirse por `latest` en contextos de actualización controlada.

## 9. Seguridad y Gobierno de Datos
La seguridad se aborda desde varias aristas. En la cadena de CI se realizan escaneos de vulnerabilidades de imágenes y dependencias. Las credenciales sensibles se almacenan como secretos de GitHub y sólo se inyectan en tiempo de ejecución de los workflows. Se mantiene un esquema de datos documentado y validaciones básicas que ayudan a prevenir incidencias por cambios silenciosos en el origen. Para futuras iteraciones, se recomienda incorporar verificaciones más estrictas de integridad y controles de acceso a los artefactos.

## 10. Pruebas, Calidad y CI/CD
Los flujos de automatización incluyen la publicación en PyPI, la construcción y el envío de la imagen Docker y la ejecución del pipeline de registro de modelos. Aunque la canalización de CI actualmente enfatiza el escaneo de seguridad para acortar tiempos de retroalimentación, el proyecto está preparado para reactivar pruebas unitarias y linters como requisito de integración. Esta práctica, junto con la política de ramas y pull requests, permitiría institucionalizar la calidad y reducir regresiones.

## 11. Costos, Viabilidad y Escalabilidad
La solución es viable en servidores de bajo coste: un Droplet con entre una y dos vCPU y un rango de 2 a 4 GB de memoria es suficiente para cargas iniciales, con un coste mensual aproximado entre cinco y doce dólares. La escalabilidad horizontal se alcanza ejecutando varias réplicas del contenedor detrás de un balanceador de carga. A medida que el uso crezca, será posible introducir orquestadores como Docker Swarm o Kubernetes y mecanismos de autoscalado. El tamaño de la imagen puede optimizarse con construcciones multi‑stage y ruedas precompiladas para acelerar despliegues.

## 12. Presentación de Negocio
Desde la perspectiva de negocio, el problema que se resuelve es la clasificación manual y repetitiva de documentos, una tarea lenta, costosa y propensa a inconsistencias. La propuesta de valor es una solución automatizada que procesa grandes volúmenes de información con tiempos de respuesta predecibles y criterios uniformes. La implantación de esta herramienta permite liberar horas de trabajo de alto valor, acelerar los tiempos de ciclo y mejorar la trazabilidad de decisiones. Con un coste de infraestructura reducido y una arquitectura estandarizada, el retorno de inversión es favorable incluso en escenarios de demanda variable. El riesgo operativo se controla con versionado, monitoreo básico y capacidad de reversión, mientras que el roadmap contempla mejoras graduales sin interrumpir el servicio.

## 13. Conclusiones
El proyecto cumple los requisitos al presentar una solución funcional y documentada que integra metodología CRISP‑DM con prácticas MLOps. La arquitectura facilita la reproducibilidad y sienta las bases para escalar tanto técnica como organizacionalmente. A corto plazo se recomienda fortalecer la observabilidad con métricas y tableros de monitoreo, habilitar pruebas automatizadas en la integración continua y estudiar la viabilidad de exponer un servicio de inferencia en tiempo real. A mediano plazo, la incorporación de representaciones más ricas y de orquestación de contenedores permitirá atender picos de demanda y nuevos casos de uso sin comprometer la calidad.

## 14. Trabajo Futuro
Las líneas de evolución incluyen la publicación de una API de inferencia basada en FastAPI, la ejecución de pruebas de carga y la activación de políticas de autoscalado. También se prevé integrar un feature store y orquestadores de datos para flujos complejos, así como un sistema de monitoreo avanzado con Prometheus y Grafana que permita alertas proactivas.

## 15. Referencias
CRISP‑DM: https://www.sv-europa.org/crisp-dm.pdf  
Scikit‑learn: https://scikit-learn.org/  
NLTK: https://www.nltk.org/  
Docker: https://docs.docker.com/  
GitHub Actions: https://docs.github.com/actions

## 16. Anexos
Acciones: https://github.com/angelcast2002/PROYECTO-MLOPS/actions  
Releases: https://github.com/angelcast2002/PROYECTO-MLOPS/releases  
PyPI: https://pypi.org/project/proyecto-mlops/  
Docker Hub: https://hub.docker.com/repository/docker/angelcast2025/proyecto-mlops/general  
Guía de despliegue en DigitalOcean: `docs_project/DEPLOY_DIGITAL_OCEAN.md`