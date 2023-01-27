import subprocess
import json
txt = "hello"
text_dict = {"text":txt}
process = subprocess.run(['curl', 'localhost:5005/model/parse', '-d', json.dumps(text_dict)],
                         stdout=subprocess.PIPE,
                         universal_newlines=True)
# process = subprocess.run(['curl', 'localhost:5005/model/parse', '-d', "{'text':{0}}".format('hello')],
#                          stdout=subprocess.PIPE,
#                          universal_newlines=True)

print(process.stdout)

# curl localhost:5005/model/parse -d '{"text":"my child get bloating"}'