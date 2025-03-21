from picarro import *





file_path='/home/jmn/Headers.xlsx'

'''
headers=header(file_path, sheet_name='stations', header=0)

print(headers.columns)

'''


site_value = 'altz'
variables = extract_variables(file_path, sheet_name='stations', header=0, site_value=site_value)
if variables:
    name, state, north, west, masl, ut = variables
    print(f"Name: {name}")
    print(f"State: {state}")
    print(f"North: {north}")
    print(f"West: {west}")
    print(f"MASL: {masl}")
    print(f"UT: {ut}")
else:
    print(f"No data found for site: {site_value}")