"""
Generador de dataset sintético: Servicios de recepción y visita técnica
de una empresa de venta/mantenimiento de equipos de fisioterapia.

Basado en la memoria y experiencia real del autor como coordinador de
servicio técnico (2 años). Nombres de marcas y clientes fueron modificados
respecto a los reales para evitar identificarlos.

Cómo ajustar el dataset:
- Cambia los "peso" dentro de EQUIPOS y CLIENTES para variar frecuencias.
- Cambia N_SERVICIOS_POR_ANIO para más o menos volumen.
- Cambia PESOS_MES para variar la estacionalidad.
- Cambia MONTO_RANGOS para otros rangos de precios.
"""

import random
import numpy as np
import pandas as pd
from datetime import date, timedelta

random.seed(42)
np.random.seed(42)

# ---------------------------------------------------------------------------
# 1. VOLUMEN POR AÑO (creciente, calibrado con las ganancias reales que diste)
# ---------------------------------------------------------------------------
N_SERVICIOS_POR_ANIO = {
    2022: 250,   # 2 personas en el área
    2023: 270,
    2024: 330,   # entra el autor como practicante, empieza digitalización
    2025: 430,   # autor como coordinador interino, usa movilidad propia
}

# Estacionalidad (pesos relativos por mes, calibrados con el gráfico de
# ganancias 2025 que describiste: bajo ene/feb/may/ago/nov, alto mar/abr/dic)
PESOS_MES = {
    1: 3, 2: 3, 3: 13, 4: 12, 5: 4, 6: 8,
    7: 8, 8: 4, 9: 8, 10: 9, 11: 4, 12: 14,
}

# Proporción Recepción vs Visita (flete solo en visita -> mucho menos frecuente)
PROB_RECEPCION = 0.80

# ---------------------------------------------------------------------------
# 2. EQUIPOS: categoría -> peso, marcas/modelos, fallas con su "trabajo realizado"
# ---------------------------------------------------------------------------
EQUIPOS = {
    "Magnetoterapia": {
        "peso": 35,
        "marcas": {
            "Meditek": ["Magnetherp 200", "Magnetherp 330", "Magnetherp 330 NG", "Magnetherp 440"],
            "Emme": ["Magnetomed 7200", "Magnetomed 8400"],
            "Storz Medikal": ["Magnetolith"],
        },
        "fallas": [
            ("Falso contacto en cable de aplicador (bobina)", 45, "Cambio de cable de aplicador", "Cable de aplicador"),
            ("Rotura de carcasa de bobina plana (orejas de correa)", 18, "Cambio de carcasa de accesorio", "Carcasa de bobina"),
            ("Descalibración", 15, "Calibración de equipo", None),
            ("Falla en teclado membrana", 8, "Cambio de teclado membrana", "Teclado membrana"),
            ("Fusible quemado", 5, "Cambio de fusibles", "Fusible"),
            ("Falla en cable de poder", 4, "Cambio de cable de poder", "Cable de poder"),
            ("Falla en placa de potencia (reparable)", 3, "Reparación de placa de potencia", None),
            ("Falla en placa de control (reparable)", 2, "Reparación de placa de control", None),
        ],
    },
    "Ultrasonido": {
        "peso": 25,
        "marcas": {
            "Carsi": ["Sonomed IV", "Sonomed V"],
            "Emme": ["Ultrasonic 1500", "Ultrasonic 1300"],
        },
        "fallas": [
            ("Falso contacto en cable de aplicador", 40, "Cambio de cable de aplicador", "Cable de aplicador US"),
            ("Descalibración", 25, "Calibración de equipo", None),
            ("Falla en teclado membrana", 12, "Cambio de teclado membrana", "Teclado membrana"),
            ("Desgaste de cristal piezoeléctrico de aplicador", 8, "Cambio de piezoeléctrico de aplicador", "Aplicador US"),
            ("Fusible quemado", 5, "Cambio de fusibles", "Fusible"),
            ("Falla en placa de potencia (reparable)", 4, "Reparación de placa de potencia", None),
            ("Falla en placa de control (reparable)", 3, "Reparación de placa de control", None),
            ("Falla en cable de poder", 3, "Cambio de cable de poder", "Cable de poder"),
        ],
    },
    "Electroterapia": {
        "peso": 20,
        "marcas": {
            "Biomedikal": ["Quadstar II", "Biostim NMS", "Impulse 3000T"],
            "Emme": ["Therapic 9200", "Therapic 9400"],
        },
        "fallas": [
            ("Falso contacto en cable de electrodos", 45, "Cambio de cable de electrodos", "Cable de electrodos"),
            ("Electrodos desgastados", 20, "Cambio de electrodos", "Electrodos"),
            ("Falla en placa de control por caída (equipo portátil)", 12, "Reparación de placa de control", None),
            ("Falla en placa de control (equipo grande)", 8, "Reparación de placa de control", None),
            ("Descalibración", 8, "Calibración de equipo", None),
            ("Falla en teclado / perillas", 4, "Cambio de teclado o perillas", "Teclado"),
            ("Fusible quemado", 3, "Cambio de fusibles", "Fusible"),
        ],
    },
    "Laser baja potencia": {
        "peso": 20,
        "marcas": {
            "Carsi": ["Lasermed 60mW", "Lasermed 100mW"],
        },
        "fallas": [
            ("Falso contacto en cable de cánula láser", 35, "Cambio de cable de cánula láser", "Cánula láser"),
            ("Descalibración", 25, "Calibración de equipo", None),
            ("Falla en teclado membrana", 18, "Cambio de teclado membrana", "Teclado membrana"),
            ("Falla en cable de poder", 6, "Cambio de cable de poder", "Cable de poder"),
            ("Fusible quemado", 5, "Cambio de fusibles", "Fusible"),
            ("Falla en placa de potencia (reparable)", 6, "Reparación de placa de potencia", None),
            ("Falla en placa de control (reparable)", 5, "Reparación de placa de control", None),
        ],
    },
    "Compreseros Calientes": {
        "peso": 7,
        "marcas": {
            "Nacional Perú": ["Compresero 6 unid.", "Compresero 12 unid.", "Compresero 24 unid."],
        },
        "fallas": [
            ("Descalibración de temperatura", 55, "Calibración de temperatura", None),
            ("Resistencia dañada", 20, "Cambio de resistencia", "Resistencia"),
            ("Termostato dañado", 15, "Cambio de termostato", "Termostato"),
            ("Fusible quemado", 6, "Cambio de fusibles", "Fusible"),
            ("Falla en cable de poder", 2, "Cambio de cable de poder", "Cable de poder"),
            ("Falla en interruptor de encendido", 2, "Cambio de interruptor de encendido", "Interruptor"),
        ],
    },
    "Dermatoscopios": {
        "peso": 7,
        "marcas": {
            "Dermlitex": ["DL5", "DL4", "DL4W", "DL200HR", "DL3", "DL100", "Lumio", "Lumio S"],
        },
        "fallas": [
            ("Mantenimiento por limpieza de lente", 60, "Limpieza y mantenimiento de lente", "Lente"),
            ("Botones dañados", 20, "Cambio de botones", "Botonera"),
            ("Batería agotada / dañada", 12, "Cambio de batería", "Batería"),
            ("Falla en placa por caída", 8, "Reparación de placa por caída", None),
        ],
    },
    "Ondas de Choque Radial": {
        "peso": 2,
        "marcas": {
            "Ztorz Medikal": ["Masterpuls MP200", "Masterpuls MP ONE"],
        },
        "fallas": [
            ("Aplicador superó límite de disparos", 90, "Cambio de aplicador por límite de disparos", "Aplicador"),
            ("Actualización de software pendiente", 6, "Actualización de software", None),
            ("Falla en cable de poder", 4, "Cambio de cable de poder", "Cable de poder"),
        ],
    },
    "Camilla de Tracción": {
        "peso": 2,
        "marcas": {
            "Everyway": ["ET-800"],
        },
        "fallas": [
            ("Descalibración de fuerza de tracción", 55, "Calibración de fuerza de tracción", None),
            ("Falla mecánica del sistema interno por mal uso", 25, "Reparación de sistema interno", None),
            ("Mantenimiento general", 20, "Mantenimiento preventivo general", None),
        ],
    },
    "Compresero Frío": {
        "peso": 1,
        "marcas": {"Nacional Perú": ["Compresero frío 6 unid.", "Compresero frío 12 unid."]},
        "fallas": [("Mantenimiento preventivo (sin falla registrada)", 100, "Mantenimiento preventivo general", None)],
    },
    "Masoterapia": {
        "peso": 1,
        "marcas": {"Genérico": ["Equipo de masoterapia"]},
        "fallas": [("Caso particular / falla no recurrente", 100, "Reparación según diagnóstico particular", None)],
    },
    "Laser alta potencia": {
        "peso": 2,
        "marcas": {"Emme": ["Bipower Lux SP", "Bipower Lux", "Vikate 8W", "Lasermed 2200"]},
        "fallas": [
            ("Descalibración", 55, "Calibración de equipo", None),
            ("Falla en cánula láser", 25, "Cambio de cánula láser", "Cánula láser"),
            ("Caso particular de fábrica", 20, "Reparación según diagnóstico particular", None),
        ],
    },
    "Lampara Infrarroja": {
        "peso": 1,
        "marcas": {"Genérico": ["Lámpara infrarroja"]},
        "fallas": [("Mantenimiento preventivo (sin falla registrada)", 100, "Mantenimiento preventivo general", None)],
    },
    "Bicicleta Estacionaria": {
        "peso": 1,
        "marcas": {"Xterra": ["SU 139", "7.0U", "7.0R"], "Ergolines": ["Ergoselect 200P", "Ergoselect 150P"]},
        "fallas": [("Armado / instalación inicial", 100, "Armado e instalación", None)],
    },
}

# Después de casi cualquier falla, hay alta probabilidad de que también se
# aproveche para dar mantenimiento preventivo anual (tal como describiste).
PROB_MTTO_PREVENTIVO_ADICIONAL = 0.55

# ---------------------------------------------------------------------------
# 3. RANGOS DE MONTO por tipo de trabajo (en la moneda del cliente)
# ---------------------------------------------------------------------------
MONTO_RANGOS = {
    "Cambio de cable de aplicador": (100, 220),
    "Cambio de cable de aplicador (bobina)": (100, 220),
    "Cambio de carcasa de accesorio": (90, 180),
    "Calibración de equipo": (100, 200),
    "Calibración de temperatura": (80, 150),
    "Calibración de fuerza de tracción": (100, 200),
    "Cambio de teclado membrana": (150, 320),
    "Cambio de fusibles": (50, 100),
    "Cambio de cable de poder": (80, 150),
    "Reparación de placa de potencia": (280, 600),
    "Reparación de placa de control": (250, 550),
    "Cambio de placa (no reparable)": (650, 1500),
    "Cambio de cable de electrodos": (90, 180),
    "Cambio de electrodos": (60, 130),
    "Cambio de piezoeléctrico de aplicador": (200, 400),
    "Cambio de cable de cánula láser": (110, 220),
    "Cambio de cánula láser": (150, 300),
    "Cambio de resistencia": (90, 180),
    "Cambio de termostato": (100, 200),
    "Cambio de interruptor de encendido": (60, 120),
    "Limpieza y mantenimiento de lente": (60, 120),
    "Cambio de botones": (70, 150),
    "Cambio de batería": (80, 160),
    "Reparación de placa por caída": (200, 450),
    "Cambio de aplicador por límite de disparos": (1500, 4200),
    "Actualización de software": (200, 400),
    "Reparación de sistema interno": (250, 500),
    "Mantenimiento preventivo general": (80, 160),
    "Cambio de teclado o perillas": (100, 220),
    "Reparación según diagnóstico particular": (150, 500),
    "Armado e instalación": (150, 350),
}
MONTO_MTTO_ADICIONAL = (60, 120)  # se suma cuando hay mantenimiento preventivo extra

# ---------------------------------------------------------------------------
# 4. CLIENTES: nombre, tipo, peso (frecuencia relativa), moneda, factor de monto
# ---------------------------------------------------------------------------
CLIENTES = [
    # (nombre, tipo, peso, moneda, factor_monto)
    ("Patricia Gómez Vásquez EIRL", "Consultorio independiente", 14, "PEN", 0.8),
    ("Mefyre - Cuerpo de Emergencias", "Institución", 11, "PEN", 1.0),
    ("Maribel Salud SAC", "Clínica", 9, "PEN", 0.9),
    ("Alfa y Omega Rehabilitación", "Clínica", 6, "PEN", 0.9),
    ("Carlos Alberto Santos", "Paciente / equipo propio", 6, "PEN", 0.6),
    ("Clínica Primavera SAC", "Clínica", 6, "PEN", 1.0),
    ("Corporación de Servicios Médicos del Sur", "Clínica", 6, "PEN", 1.0),
    ("NoPain Centro de Rehabilitación", "Clínica", 6, "PEN", 0.9),
    ("Parroquia Nuestra Señora de Chaclacayo", "Institución", 6, "PEN", 0.9),
    ("Seguro Social de Salud", "Institución (VIP)", 3, "PEN", 6.0),
    ("IAFAS Fuerza Aeroespacial del Perú", "Institución (VIP)", 3, "PEN", 4.5),
    ("Federación Peruana de Voleibol", "Institución (VIP)", 3, "PEN", 4.3),
    ("Trauma Médika EIRL", "Clínica (VIP)", 3, "USD", 3.5),
    ("Universidad Nacional del Centro", "Institución", 2, "PEN", 2.0),
    ("Hospital Nacional Docente", "Institución", 2, "PEN", 1.7),
    # cola larga de clientes menos frecuentes
    ("Fisioterapia San Rafael", "Consultorio independiente", 3, "PEN", 0.8),
    ("Lic. Mónica Torres Fisioterapia", "Consultorio independiente", 3, "PEN", 0.7),
    ("Clínica Los Álamos SAC", "Clínica", 3, "PEN", 1.0),
    ("Centro de Rehabilitación Vitalis", "Clínica", 3, "PEN", 0.9),
    ("Lic. Jorge Injante Terapia Física", "Consultorio independiente", 2, "PEN", 0.7),
    ("Instituto Peruano del Deporte - Sede Sur", "Institución", 2, "PEN", 2.2),
    ("Clínica Bienestar Total EIRL", "Clínica", 2, "PEN", 1.0),
    ("Policlínico San Judas", "Clínica", 2, "PEN", 0.9),
    ("Fisiosalud Andina SAC", "Clínica", 2, "PEN", 0.9),
    ("Lic. Rosa Ibarra Terapia Física", "Consultorio independiente", 2, "PEN", 0.7),
    ("Centro Médico El Progreso", "Clínica", 2, "PEN", 1.0),
]

# Direcciones / contacto ficticio por cliente (fijo por cliente)
DISTRITOS = ["Chaclacayo", "Ate", "San Juan de Lurigancho", "La Molina", "Surco",
             "San Borja", "San Isidro", "Miraflores", "Los Olivos", "San Miguel",
             "Cercado de Lima", "Chosica", "Comas", "Jesús María", "Magdalena"]

def generar_contacto(nombre_cliente, idx):
    distrito = random.choice(DISTRITOS)
    telefono = f"9{random.randint(10000000, 99999999)}"
    dominio = random.choice(["gmail.com", "hotmail.com", "outlook.com"])
    slug = "".join(c for c in nombre_cliente.lower() if c.isalnum())[:12]
    correo = f"{slug}{idx}@{dominio}"
    direccion = f"Av. {random.choice(['Los Próceres','Las Torres','El Sol','Central','Las Flores','La Marina'])} {random.randint(100,1999)}, {distrito}"
    contacto = random.choice(["Recepción", "Administración", "Coordinación", "Encargado de compras"])
    return telefono, correo, direccion, distrito, contacto

CLIENTES_INFO = {}
for i, (nombre, tipo, peso, moneda, factor) in enumerate(CLIENTES):
    tel, correo, direccion, distrito, contacto = generar_contacto(nombre, i)
    CLIENTES_INFO[nombre] = {
        "tipo": tipo, "peso": peso, "moneda": moneda, "factor": factor,
        "telefono": tel, "correo": correo, "direccion": direccion,
        "distrito": distrito, "contacto": contacto,
    }

# ---------------------------------------------------------------------------
# 5. GENERACIÓN
# ---------------------------------------------------------------------------
TECNICOS_POR_ANIO = {
    2022: [("Coordinador Técnico", 0.85), ("Practicante", 0.15)],
    2023: [("Coordinador Técnico", 0.85), ("Practicante", 0.15)],
    2024: [("Encargado de Servicio Técnico", 0.65), ("Coordinador Técnico", 0.20), ("Practicante", 0.15)],
    2025: [("Encargado de Servicio Técnico", 0.75), ("Practicante", 0.25)],
}

def elegir_ponderado(opciones_pesos):
    opciones, pesos = zip(*opciones_pesos)
    return random.choices(opciones, weights=pesos, k=1)[0]

def elegir_categoria_equipo():
    cats = list(EQUIPOS.keys())
    pesos = [EQUIPOS[c]["peso"] for c in cats]
    return random.choices(cats, weights=pesos, k=1)[0]

def elegir_marca_modelo(categoria):
    marcas = EQUIPOS[categoria]["marcas"]
    marca = random.choice(list(marcas.keys()))
    modelo = random.choice(marcas[marca])
    return marca, modelo

def elegir_falla(categoria):
    fallas = EQUIPOS[categoria]["fallas"]
    descripciones, pesos, trabajos, accesorios = zip(*fallas)
    idx = random.choices(range(len(fallas)), weights=pesos, k=1)[0]
    return descripciones[idx], trabajos[idx], accesorios[idx]

def elegir_cliente():
    nombres = [c[0] for c in CLIENTES]
    pesos = [c[2] for c in CLIENTES]
    return random.choices(nombres, weights=pesos, k=1)[0]

def fecha_aleatoria(anio):
    mes = random.choices(list(PESOS_MES.keys()), weights=list(PESOS_MES.values()), k=1)[0]
    dias_en_mes = {1:31,2:28,3:31,4:30,5:31,6:30,7:31,8:31,9:30,10:31,11:30,12:31}[mes]
    dia = random.randint(1, dias_en_mes)
    return date(anio, mes, dia)

def calcular_monto(trabajo, monto_extra_mtto, factor_cliente):
    lo, hi = MONTO_RANGOS.get(trabajo, (80, 200))
    base = random.uniform(lo, hi)
    if monto_extra_mtto:
        lo2, hi2 = MONTO_MTTO_ADICIONAL
        base += random.uniform(lo2, hi2)
    return round(base * factor_cliente, 2)

filas = []
n_cot = {a: 0 for a in N_SERVICIOS_POR_ANIO}
n_inf = {a: 0 for a in N_SERVICIOS_POR_ANIO}

for anio, n_servicios in N_SERVICIOS_POR_ANIO.items():
    for _ in range(n_servicios):
        tipo_servicio = "Recepción" if random.random() < PROB_RECEPCION else "Visita"
        categoria = elegir_categoria_equipo()
        marca, modelo = elegir_marca_modelo(categoria)
        falla_desc, trabajo, accesorio = elegir_falla(categoria)

        mtto_extra = (trabajo != "Mantenimiento preventivo general") and (random.random() < PROB_MTTO_PREVENTIVO_ADICIONAL)
        trabajo_realizado = trabajo
        if mtto_extra:
            trabajo_realizado += " + Mantenimiento preventivo anual"

        cliente = elegir_cliente()
        info_cliente = CLIENTES_INFO[cliente]

        fecha_ingreso = fecha_aleatoria(anio)
        dias_resolucion = max(1, int(np.random.gamma(shape=2.2, scale=2.3)))  # cola larga realista
        fecha_entrega = fecha_ingreso + timedelta(days=dias_resolucion)

        n_cot[anio] += 1
        n_inf[anio] += 1
        num_cotizacion = f"COT-{anio}-{n_cot[anio]:04d}"
        num_informe = f"INF-{anio}-{n_inf[anio]:04d}"

        monto = calcular_monto(trabajo, mtto_extra, info_cliente["factor"])
        tecnico = elegir_ponderado(TECNICOS_POR_ANIO[anio])

        filas.append({
            "anio": anio,
            "tipo_servicio": tipo_servicio,
            "fecha_ingreso": fecha_ingreso,
            "fecha_entrega": fecha_entrega,
            "dias_resolucion": dias_resolucion,
            "razon_social": cliente,
            "tipo_cliente": info_cliente["tipo"],
            "distrito": info_cliente["distrito"],
            "contacto": info_cliente["contacto"],
            "telefono": info_cliente["telefono"],
            "correo": info_cliente["correo"],
            "direccion": info_cliente["direccion"],
            "categoria_equipo": categoria,
            "marca": marca,
            "modelo": modelo,
            "n_serie": f"{marca[:3].upper()}-{random.randint(1000,9999)}",
            "accesorio_afectado": accesorio if accesorio else "N/A",
            "descripcion_falla": falla_desc,
            "trabajo_realizado": trabajo_realizado,
            "mantenimiento_preventivo_adicional": "Sí" if mtto_extra else "No",
            "observaciones": random.choice([
                "Equipo con uso intensivo diario", "Cliente solicitó revisión urgente",
                "Equipo antiguo, se recomienda mantenimiento anual", "Sin observaciones adicionales",
                "Cliente rechazó cambio de placa, se optó por reparación",
                "Se coordinó con el cliente por retraso de repuesto importado", "",
            ]),
            "tecnico_encargado": tecnico,
            "n_cotizacion": num_cotizacion,
            "n_informe_tecnico": num_informe,
            "moneda": info_cliente["moneda"],
            "monto_servicio": monto,
        })

df = pd.DataFrame(filas)
df = df.sort_values("fecha_ingreso").reset_index(drop=True)

# ---------------------------------------------------------------------------
# 6. EXPORTAR
# ---------------------------------------------------------------------------
df.to_csv("servicios_tecnicos_fisioterapia.csv", index=False, encoding="utf-8-sig")

with pd.ExcelWriter("servicios_tecnicos_fisioterapia.xlsx", engine="openpyxl") as writer:
    df.to_excel(writer, sheet_name="Todos los servicios", index=False)
    df[df["tipo_servicio"] == "Recepción"].to_excel(writer, sheet_name="Recepción", index=False)
    df[df["tipo_servicio"] == "Visita"].to_excel(writer, sheet_name="Visita", index=False)

print(f"Total de filas generadas: {len(df)}")
resumen = df.groupby(["anio", "moneda"])["monto_servicio"].agg(["count", "sum"])
resumen.columns = ["servicios", "total_monto"]
print(resumen)
