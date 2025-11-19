import pandas as pd
import numpy as np

#cuantos semanas hay del 15 de octubre de 2024 al 17 de abril de 2025

start_date = pd.to_datetime('2024-10-15')
end_date = pd.to_datetime('2025-04-17')

# Calcular días laborales (lunes a viernes)
business_days = pd.bdate_range(start=start_date, end=end_date)
total_workdays = len(business_days)

# Calcular semanas laborales
workweeks = total_workdays / 5

# Crear contenido para el archivo
contenido = []
contenido.append(f"Fecha inicio: {start_date.strftime('%d-%m-%Y')}")
contenido.append(f"Fecha fin: {end_date.strftime('%d-%m-%Y')}")
contenido.append(f"Días laborales totales: {total_workdays}")
contenido.append(f"Semanas laborales: {workweeks:.1f}")
contenido.append("\nDetalle por semanas:")

print(f"Fecha inicio: {start_date.strftime('%d-%m-%Y')}")
print(f"Fecha fin: {end_date.strftime('%d-%m-%Y')}")
print(f"Días laborales totales: {total_workdays}")
print(f"Semanas laborales: {workweeks:.1f}")
print("\nDetalle por semanas:")

# Mostrar cada semana laboral
current_date = start_date
semana = 1

while current_date <= end_date:
    # Encontrar el inicio de la semana laboral
    # Si es sábado o domingo, mover al lunes siguiente
    if current_date.weekday() == 5:  # Sábado
        fecha_inicial_semana = current_date + pd.Timedelta(days=2)
    elif current_date.weekday() == 6:  # Domingo
        fecha_inicial_semana = current_date + pd.Timedelta(days=1)
    else:
        fecha_inicial_semana = current_date
    
    # Si la fecha inicial ya pasó el end_date, terminar
    if fecha_inicial_semana > end_date:
        break
    
    # Calcular el final de la semana laboral (viernes)
    days_to_friday = (4 - fecha_inicial_semana.weekday()) % 7
    if fecha_inicial_semana.weekday() > 4:  # Si es sábado o domingo
        days_to_friday = 4
    fecha_final_semana = fecha_inicial_semana + pd.Timedelta(days=days_to_friday)
    
    # Si el final de semana pasa del end_date, ajustar
    if fecha_final_semana > end_date:
        fecha_final_semana = end_date
    
    linea = f"Semana {semana}: {fecha_inicial_semana.strftime('%d-%m-%Y')} al {fecha_final_semana.strftime('%d-%m-%Y')}"
    contenido.append(linea)
    print(linea)
    
    # Mover al lunes de la siguiente semana
    next_monday = fecha_final_semana + pd.Timedelta(days=3)
    current_date = next_monday
    semana += 1

# Guardar en archivo de texto
with open('semanas_hora.txt', 'w', encoding='utf-8') as f:
    for linea in contenido:
        f.write(linea + '\n')

print(f"\nArchivo 'semanas_hora.txt' guardado exitosamente.")


# horas salvadas  
'''

dias inhabiles 

1 noviembre 2024
2 noviembre 2024
18 de noviembre 2024

12 diciembre 2024

16-dicembre 2024 al 3 de enero 2025

3 de febrero 2025
17 marzo 2025 
14 al 18 de abril 2025
'''

