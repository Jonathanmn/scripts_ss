import pandas as pd
import numpy as np

#cuantos semanas hay del 15 de octubre de 2024 al 17 de abril de 2025

start_date = pd.to_datetime('2024-10-15')
end_date = pd.to_datetime('2025-04-17')

# Definir días inhábiles
dias_inhabiles = [
    pd.to_datetime('2024-11-01'),
    pd.to_datetime('2024-11-02'),
    pd.to_datetime('2024-11-18'),
    pd.to_datetime('2024-12-12'),
    # Vacaciones de diciembre
    *pd.date_range('2024-12-16', '2025-01-03'),
    pd.to_datetime('2025-02-03'),
    pd.to_datetime('2025-03-17'),
    # Vacaciones de abril
    *pd.date_range('2025-04-14', '2025-04-18')
]

# Función para contar días laborales válidos en una semana
def contar_dias_laborales_validos(fecha_inicio, fecha_fin):
    dias_semana = pd.date_range(start=fecha_inicio, end=fecha_fin, freq='D')
    dias_validos = 0
    for dia in dias_semana:
        # Solo contar si es día laboral (lunes a viernes) y no es inhábil
        if dia.weekday() < 5 and dia not in dias_inhabiles:
            dias_validos += 1
    return dias_validos

# Crear contenido para el archivo con formato de columnas
contenido = []
contenido.append(f"Fecha inicio: {start_date.strftime('%d-%m-%Y')}")
contenido.append(f"Fecha fin inicial: {end_date.strftime('%d-%m-%Y')}")
contenido.append(f"Meta: 480 horas totales")
contenido.append("")
contenido.append("Semana\t\tTabla\t\tFecha\t\t\t\tHoras Semana\tHoras Totales")
contenido.append("-" * 80)

print(f"Fecha inicio: {start_date.strftime('%d-%m-%Y')}")
print(f"Fecha fin inicial: {end_date.strftime('%d-%m-%Y')}")
print(f"Meta: 480 horas totales")
print()
print("Semana\t\tTabla\t\tFecha\t\t\t\tHoras Semana\tHoras Totales")
print("-" * 80)

# Mostrar cada semana laboral
current_date = start_date
semana = 1
total_horas_acumuladas = 0
dias_inhabiles_por_semana = {}  # Diccionario para almacenar días inhábiles por semana

while total_horas_acumuladas < 480:
    # Encontrar el inicio de la semana laboral
    if current_date.weekday() == 5:  # Sábado
        fecha_inicial_semana = current_date + pd.Timedelta(days=2)
    elif current_date.weekday() == 6:  # Domingo
        fecha_inicial_semana = current_date + pd.Timedelta(days=1)
    else:
        fecha_inicial_semana = current_date
    
    # Calcular el final de la semana laboral (viernes)
    days_to_friday = (4 - fecha_inicial_semana.weekday()) % 7
    if fecha_inicial_semana.weekday() > 4:
        days_to_friday = 4
    fecha_final_semana = fecha_inicial_semana + pd.Timedelta(days=days_to_friday)
    
    # Verificar qué días inhábiles caen en esta semana
    for dia_inhabil in dias_inhabiles:
        if fecha_inicial_semana <= dia_inhabil <= fecha_final_semana:
            if dia_inhabil not in dias_inhabiles_por_semana:
                dias_inhabiles_por_semana[dia_inhabil] = semana
    
    # Contar días laborales válidos (excluyendo inhábiles)
    dias_validos = contar_dias_laborales_validos(fecha_inicial_semana, fecha_final_semana)
    horas_semana = dias_validos * 4
    total_horas_acumuladas += horas_semana
    
    # Calcular número de tabla (cada 4 semanas)
    tabla = ((semana - 1) // 4) + 1
    
    # Formato de fecha para la columna
    fecha_rango = f"{fecha_inicial_semana.strftime('%d-%m-%Y')} al {fecha_final_semana.strftime('%d-%m-%Y')}"
    
    linea = f"{semana}\t\t{tabla}\t\t{fecha_rango}\t\t{horas_semana}h\t\t{total_horas_acumuladas}h"
    contenido.append(linea)
    print(linea)
    
    # Si ya llegamos a 480 horas, terminar
    if total_horas_acumuladas >= 480:
        break
    
    # Mover al lunes de la siguiente semana
    next_monday = fecha_final_semana + pd.Timedelta(days=3)
    current_date = next_monday
    semana += 1

# Agregar resumen final
contenido.append("")
contenido.append("=" * 80)
contenido.append("RESUMEN FINAL")
contenido.append("=" * 80)
contenido.append(f"Total de semanas: {semana}")
contenido.append(f"Fecha final para completar 480h: {fecha_final_semana.strftime('%d-%m-%Y')}")
contenido.append(f"Total de horas: {total_horas_acumuladas}h")
contenido.append("")
contenido.append("DÍAS INHÁBILES:")
for dia in sorted(dias_inhabiles):
    semana_encontrada = dias_inhabiles_por_semana.get(dia, "No incluida")
    dia_semana = dia.strftime('%A')  # Obtener día de la semana en inglés
    # Traducir a español
    dias_es = {
        'Monday': 'Lunes',
        'Tuesday': 'Martes', 
        'Wednesday': 'Miércoles',
        'Thursday': 'Jueves',
        'Friday': 'Viernes',
        'Saturday': 'Sábado',
        'Sunday': 'Domingo'
    }
    dia_semana_es = dias_es.get(dia_semana, dia_semana)
    
    if semana_encontrada != "No incluida":
        contenido.append(f"- {dia.strftime('%d-%m-%Y')} ({dia_semana_es}) - Semana {semana_encontrada}")
    else:
        contenido.append(f"- {dia.strftime('%d-%m-%Y')} ({dia_semana_es}) - No incluida en el período")

print()
print("=" * 80)
print("RESUMEN FINAL")
print("=" * 80)
print(f"Total de semanas: {semana}")
print(f"Fecha final para completar 480h: {fecha_final_semana.strftime('%d-%m-%Y')}")
print(f"Total de horas: {total_horas_acumuladas}h")
print()
print("DÍAS INHÁBILES:")
for dia in sorted(dias_inhabiles):
    semana_encontrada = dias_inhabiles_por_semana.get(dia, "No incluida")
    dia_semana = dia.strftime('%A')  # Obtener día de la semana en inglés
    # Traducir a español
    dias_es = {
        'Monday': 'Lunes',
        'Tuesday': 'Martes', 
        'Wednesday': 'Miércoles',
        'Thursday': 'Jueves',
        'Friday': 'Viernes',
        'Saturday': 'Sábado',
        'Sunday': 'Domingo'
    }
    dia_semana_es = dias_es.get(dia_semana, dia_semana)
    
    if semana_encontrada != "No incluida":
        print(f"- {dia.strftime('%d-%m-%Y')} ({dia_semana_es}) - Semana {semana_encontrada}")
    else:
        print(f"- {dia.strftime('%d-%m-%Y')} ({dia_semana_es}) - No incluida en el período")

# Guardar en archivo de texto
with open('informe/horas_totales.txt', 'w', encoding='utf-8') as f:
    for linea in contenido:
        f.write(linea + '\n')

print(f"\nArchivo 'horas_totales.txt' guardado exitosamente.")