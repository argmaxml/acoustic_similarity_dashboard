from streamlit import bootstrap
# debugging tool
real_script = 'app.py'

bootstrap.run(real_script, f'run.py {real_script}', [])