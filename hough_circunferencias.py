
"""TP4 - Inteligencia Artificial - Transformada de Hough para circunferencias.
Ejecutar: python hough_circunferencias.py
Requiere: numpy, matplotlib.
El algoritmo de votacion esta implementado sin usar funciones Hough de OpenCV.
"""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

ANCHO, ALTO = 240, 180
RADIO = 28
CENTRO_ESPERADO = (120, 90)
UMBRAL_VOTOS = 65
SALIDA = Path(__file__).resolve().parent / "resultados_circunferencias"


def crear_imagen(centro=None, falta_arco=False, ruido=0, semilla=2026):
    """Crea una matriz binaria de bordes, no una fotografia real."""
    imagen = np.zeros((ALTO, ANCHO), dtype=np.uint8)
    rng = np.random.default_rng(semilla)

    if centro is not None:
        cx, cy = centro
        for angulo in np.deg2rad(np.arange(360)):
            grados = np.rad2deg(angulo)
            if falta_arco and 30 <= grados < 110:
                continue

            x = round(cx + RADIO * np.cos(angulo))
            y = round(cy + RADIO * np.sin(angulo))
            if 0 <= x < ANCHO and 0 <= y < ALTO:
                imagen[y, x] = 1

    if ruido:
        xs = rng.integers(0, ANCHO, size=ruido)
        ys = rng.integers(0, ALTO, size=ruido)
        imagen[ys, xs] = 1

    return imagen


def detectar_centro(imagen, radio=RADIO, umbral=UMBRAL_VOTOS):
    """Hough circular para radio conocido: cada borde vota centros posibles."""
    acumulador = np.zeros((ALTO, ANCHO), dtype=np.int32)
    angulos = np.deg2rad(np.arange(360))

    # Evita duplicar votos cuando distintos angulos redondean al mismo offset.
    desplazamientos = np.unique(
        np.column_stack((
            np.rint(radio * np.cos(angulos)).astype(int),
            np.rint(radio * np.sin(angulos)).astype(int)
        )),
        axis=0,
    )

    ys, xs = np.nonzero(imagen)

    for x, y in zip(xs, ys):
        candidatos = np.column_stack((
            x - desplazamientos[:, 0],
            y - desplazamientos[:, 1]
        ))

        validos = (
            (candidatos[:, 0] >= 0) &
            (candidatos[:, 0] < ANCHO) &
            (candidatos[:, 1] >= 0) &
            (candidatos[:, 1] < ALTO)
        )

        candidatos = candidatos[validos]

        np.add.at(
            acumulador,
            (candidatos[:, 1], candidatos[:, 0]),
            1
        )

    mejor_y, mejor_x = np.unravel_index(
        acumulador.argmax(), acumulador.shape
    )

    votos = int(acumulador[mejor_y, mejor_x])
    centro = (
        (int(mejor_x), int(mejor_y))
        if votos >= umbral else None
    )

    return centro, votos, acumulador


def graficar(imagen, centro, votos, acumulador, titulo, archivo):
    """Crea figuras de tres paneles en español, legibles para Word."""
    fig, ejes = plt.subplots(1, 3, figsize=(15, 5.2))

    fig.subplots_adjust(
        left=0.055,
        right=0.99,
        top=0.77,
        bottom=0.18,
        wspace=0.27
    )

    ejes[0].imshow(
        imagen, cmap="gray",
        vmin=0, vmax=1,
        interpolation="nearest"
    )
    ejes[0].set_title(
        "A. Imagen de entrada",
        fontsize=12, pad=14, fontweight="bold"
    )

    ejes[1].imshow(
        acumulador,
        cmap="inferno",
        interpolation="nearest"
    )
    ejes[1].set_title(
        "B. Acumulador de votos",
        fontsize=12, pad=14, fontweight="bold"
    )

    ejes[2].imshow(
        imagen, cmap="gray",
        vmin=0, vmax=1,
        interpolation="nearest"
    )
    ejes[2].set_title(
        "C. Resultado de la detección",
        fontsize=12, pad=14, fontweight="bold"
    )

    if centro is not None:
        cx, cy = centro

        ejes[2].plot(
            cx, cy,
            marker="+", color="lime",
            markersize=13,
            markeredgewidth=2.4
        )

        ejes[2].add_patch(
            plt.Circle(
                centro, RADIO,
                fill=False,
                color="cyan",
                linewidth=1.8
            )
        )

        detalle = (
            f"Centro detectado: ({cx}, {cy}) píxeles"
            f"  |  Votos: {votos}"
        )
    else:
        detalle = (
            f"No se detectó una circunferencia"
            f"  |  Máximo: {votos} votos"
            f" (umbral: {UMBRAL_VOTOS})"
        )

    for eje in ejes:
        eje.set_xlabel(
            "Eje X (píxeles)", fontsize=10, labelpad=6
        )
        eje.set_ylabel(
            "Eje Y (píxeles)", fontsize=10, labelpad=6
        )
        eje.tick_params(axis="both", labelsize=9)

    fig.suptitle(
        titulo,
        fontsize=16,
        fontweight="bold",
        y=0.98
    )

    fig.text(
        0.5, 0.85, detalle,
        ha="center", va="center",
        fontsize=11
    )

    if centro is not None:
        fig.text(
            0.5, 0.035,
            "La cruz verde indica el centro detectado; "
            "la circunferencia celeste representa el radio conocido.",
            ha="center",
            fontsize=9,
            color="#555555"
        )

    SALIDA.mkdir(parents=True, exist_ok=True)
    destino = SALIDA / archivo

    fig.savefig(
        destino,
        dpi=180,
        facecolor="white"
    )
    plt.close(fig)

    print(f"  Gráfico: {destino}")


def main():
    desplazado = (132, 82)  # +12 pixeles en X, -8 en Y

    casos = [
        (
            "Caso 1. Aro en la posición original",
            crear_imagen(CENTRO_ESPERADO),
            CENTRO_ESPERADO,
            "01_aro_original.png"
        ),
        (
            "Caso 2. Aro desplazado",
            crear_imagen(desplazado),
            desplazado,
            "02_aro_desplazado.png"
        ),
        (
            "Caso 3. Aro desplazado con contorno incompleto y ruido",
            crear_imagen(desplazado, True, 150),
            desplazado,
            "03_aro_ruido.png"
        ),
        (
            "Caso 4. Imagen con solamente ruido",
            crear_imagen(None, ruido=150),
            None,
            "04_solo_ruido.png"
        ),
    ]

    for titulo, imagen, real, archivo in casos:
        centro, votos, acumulador = detectar_centro(imagen)

        print(f"\n{titulo}")
        print(f"  Pixeles activos: {np.count_nonzero(imagen)}")
        print(f"  Centro detectado: {centro}; votos maximos: {votos}")

        if centro is not None:
            dx = centro[0] - CENTRO_ESPERADO[0]
            dy = centro[1] - CENTRO_ESPERADO[1]

            error = (
                np.hypot(
                    centro[0] - real[0],
                    centro[1] - real[1]
                ) if real else None
            )

            print(
                f"  Desplazamiento respecto al esperado: "
                f"dx={dx}, dy={dy} pixeles"
            )
            print(f"  Error de localizacion: {error:.2f} pixeles")

        graficar(
            imagen, centro, votos,
            acumulador, titulo, archivo
        )


if __name__ == "__main__":
    main()
