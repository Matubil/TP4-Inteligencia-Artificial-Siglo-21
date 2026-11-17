"""TP4 - IA: prototipo educativo de Hough para rectas.
Ejecutar: python hough_rectas.py
Requisitos: numpy, matplotlib. La votacion de Hough se implementa manualmente.
"""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

ANCHO, ALTO = 320, 240
CENTRO = (160, 120)
ANCHO_PIEZA, ALTO_PIEZA = 170, 90
ROTACION_GRADOS = 10
THETAS = np.deg2rad(np.arange(180))
COSENOS, SENOS = np.cos(THETAS), np.sin(THETAS)
SALIDA = Path(__file__).resolve().parent


def crear_pieza(porcentaje_faltante=0, ruido=0, semilla=2026):
    """Construye un rectangulo rotado en una imagen binaria de bordes."""
    rng = np.random.default_rng(semilla)
    imagen = np.zeros((ALTO, ANCHO), dtype=np.uint8)
    angulo = np.deg2rad(ROTACION_GRADOS)
    c, s = np.cos(angulo), np.sin(angulo)
    puntos = []
    # Muestreamos los cuatro lados con resolucion subpixel.
    xs = np.linspace(-ANCHO_PIEZA / 2, ANCHO_PIEZA / 2, 1000)
    ys = np.linspace(-ALTO_PIEZA / 2, ALTO_PIEZA / 2, 700)
    for x in xs:
        puntos.extend([(x, -ALTO_PIEZA / 2), (x, ALTO_PIEZA / 2)])
    for y in ys:
        puntos.extend([(-ANCHO_PIEZA / 2, y), (ANCHO_PIEZA / 2, y)])
    for x, y in puntos:
        xr = int(round(CENTRO[0] + x*c - y*s))
        yr = int(round(CENTRO[1] + x*s + y*c))
        if 0 <= xr < ANCHO and 0 <= yr < ALTO:
            imagen[yr, xr] = 1
    coords = np.argwhere(imagen > 0)
    cantidad_quitar = int(len(coords)*porcentaje_faltante)
    if cantidad_quitar:
        seleccion = rng.choice(len(coords), size=cantidad_quitar, replace=False)
        imagen[coords[seleccion,0], coords[seleccion,1]] = 0
    if ruido:
        rx = rng.integers(0, ANCHO, size=ruido)
        ry = rng.integers(0, ALTO, size=ruido)
        imagen[ry, rx] = 1
    return imagen


def detectar_rectas(imagen, umbral=38, max_rectas=4):
    """Votacion polar manual: rho=x*cos(theta)+y*sen(theta)."""
    diagonal = int(np.ceil(np.hypot(ALTO, ANCHO)))
    acumulador = np.zeros((2*diagonal+1, 180), dtype=np.int32)
    ys, xs = np.nonzero(imagen)
    angulos = np.arange(180)
    for x,y in zip(xs, ys):
        rhos = np.rint(x*COSENOS + y*SENOS).astype(int)
        acumulador[rhos+diagonal, angulos] += 1
    temporal = acumulador.copy()
    encontrados = []
    for _ in range(max_rectas):
        fila, columna = np.unravel_index(np.argmax(temporal), temporal.shape)
        votos = int(temporal[fila,columna])
        if votos < umbral:
            break
        rho, theta = int(fila-diagonal), int(columna)
        encontrados.append((rho,theta,votos))
        # Supresion local de maximos: evita reportar el mismo borde varias veces.
        for dt in range(-7,8):
            t = (theta+dt) % 180
            # La parametrizacion polar es periodica: al cruzar 0/180 cambia signo rho.
            r = -rho if theta+dt < 0 or theta+dt >= 180 else rho
            lo = max(0,r+diagonal-12)
            hi = min(temporal.shape[0],r+diagonal+13)
            temporal[lo:hi,t]=0
    return encontrados, acumulador


def graficar(imagen, detectadas, acumulador, nombre, titulo):
    fig, ejes = plt.subplots(1,3, figsize=(13.8,4.2), constrained_layout=True)
    ejes[0].imshow(imagen,cmap='gray',origin='upper',vmin=0,vmax=1)
    ejes[0].set_title('1. Bordes de entrada')
    ejes[1].imshow(acumulador,aspect='auto',origin='lower',extent=[0,179,-400,400],cmap='viridis')
    ejes[1].set_title('2. Acumulador de Hough')
    ejes[1].set_xlabel('θ (grados)'); ejes[1].set_ylabel('ρ (píxeles)')
    ejes[2].imshow(imagen,cmap='gray',origin='upper',vmin=0,vmax=1)
    for indice,(rho,theta,votos) in enumerate(detectadas,1):
        th = np.deg2rad(theta)
        if abs(np.sin(th)) > .25:
            xx = np.array([0,ANCHO-1]); yy=(rho-xx*np.cos(th))/np.sin(th)
        else:
            yy = np.array([0,ALTO-1]); xx=(rho-yy*np.sin(th))/np.cos(th)
        ejes[2].plot(xx,yy,linewidth=1.5,label=f'R{indice}: θ={theta}°, {votos} votos')
    ejes[2].set_title('3. Rectas detectadas')
    if detectadas: ejes[2].legend(loc='upper right',fontsize=7,framealpha=.88)
    for i in (0,2):
        ejes[i].set_xlim(0,ANCHO-1);ejes[i].set_ylim(ALTO-1,0)
        ejes[i].set_xlabel('X (píxeles)');ejes[i].set_ylabel('Y (píxeles)')
    fig.suptitle(titulo,fontsize=13)
    fig.savefig(nombre, dpi=170)
    plt.close(fig)


def main():
    casos = [
        ('Caso 1: contorno completo',crear_pieza(), '01_rectas_completas.png'),
        ('Caso 2: 25 % faltante y 220 puntos de ruido',crear_pieza(.25,220),'02_rectas_ruido.png'),
        ('Caso 3: solo ruido',np.zeros((ALTO,ANCHO),dtype=np.uint8),'03_rectas_solo_ruido.png')
    ]
    rng=np.random.default_rng(2027)
    ys=rng.integers(0,ALTO,220);xs=rng.integers(0,ANCHO,220)
    casos[-1][1][ys,xs]=1
    for titulo,imagen,archivo in casos:
        rectas,acumulador=detectar_rectas(imagen)
        print('\n'+titulo)
        print(f'Pixeles activos: {np.count_nonzero(imagen)} / Rectas detectadas: {len(rectas)}')
        for n,(rho,theta,votos) in enumerate(rectas,1):
            print(f'  R{n}: rho={rho} px, theta={theta} grados, votos={votos}')
        graficar(imagen,rectas,acumulador,archivo,titulo)

if __name__=='__main__':
    main()
