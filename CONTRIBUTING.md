# Contributing Guidelines

## Cómo Contribuir

¡Gracias por tu interés en contribuir a PROYECTO-MLOPS!

### Reporte de Bugs

1. Abre un issue en GitHub
2. Describe el problema claramente
3. Incluye pasos para reproducirlo
4. Especifica tu entorno (OS, Python version, etc.)

### Propuestas de Características

1. Abre un issue con la etiqueta "enhancement"
2. Describe el caso de uso
3. Explica cómo beneficia al proyecto

### Pull Requests

1. Fork el repositorio
2. Crea una rama: `git checkout -b feature/nombre-feature`
3. Commit cambios: `git commit -am 'Describe cambios'`
4. Push: `git push origin feature/nombre-feature`
5. Abre un Pull Request

### Guía de Código

- **Estilo:** Seguir PEP 8 (usamos `black` y `isort`)
- **Tests:** Añadir tests para nuevas funcionalidades
- **Documentación:** Documentar funciones y módulos
- **Mensajes de commit:** Usar presente imperativo ("Add feature" no "Added feature")

### Ejecutar Tests Locales

```bash
pip install -e ".[dev]"
make lint      # Verificar estilo
make test      # Ejecutar tests
make format    # Formatear código
```

### Proceso de Revisión

1. Verificamos que pasen los tests en CI
2. Revisamos el código por calidad y seguridad
3. Pedimos cambios si es necesario
4. Mergeamos cuando todo esté bien

---

**¡Apreciamos todas las contribuciones!**
