# Quick setup

Guía rápida para configurar el entorno del proyecto y ejecutar los notebooks

Requisitos previos
- Git instalado.
- `uv` instalado y disponible en PATH.

Pasos
1) Clonar el repositorio y acceder al directorio:

```bash
git clone <URL_DEL_REPO>
cd si_proyecto_u3
```

2) Sincronizar las dependencias definidas en `pyproject.toml`:

```bash
uv sync
```

4) Comprobar el estado del entorno:

```bash
uv list
```

## Uso de Git en el Repositorio
1. Añadir cambios y crear commit:

```bash
git add .
git commit -m "Descripción concisa de los cambios"
```

2. Enviar al remoto:

```bash
git push origin main
```

Notas
- Este repositorio utiliza exclusivamente `uv` para la gestión de entornos y dependencias; no se emplean otros gestores en el flujo oficial.
- Si `uv sync` falla, revise que la versión de Python instalada por `uv` cumpla la restricción de `pyproject.toml` y que tenga conectividad para descargar paquetes.
