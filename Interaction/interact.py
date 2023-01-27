'''
ASSUMPTIONS - VERY VERY IMPORTANT
1) CURRENTLY I AM THINKING THAT THE QUESTIONS WHICH NEED TO BE ASKED WOULD BE PREDICTED IN ONE GO AND RETURNED AS A LIST.
2) SECONDLY IF WE WANT TO ASK MORE THEN TWO SYMPTOM IN ONE TIME THEN PLEASE RETURN THEM AS AND SEPRATED.

very important way of giving reply to the symptoms asked by the bot

* every symptom must be preceded by yes or no and order of symptoms could be any (done)
* he could give yes or no with keeping in mind the indexing of the symptom asked!!! (done)
* he could write yes all, no all, or yes and no for all type of things (think)
'''


import subprocess
import json
import random


PersonSuffering = "user"
question_asking_templates = ["was there any symptom like {}?", "Did you noticed this {}", "Did {0} showed {1}?", "Did {0} faced this {1}"]
user_symptoms = {}

def give_greetings_reply():
    return "Hey! How can I help you?"

def give_bye_reply():
    return "Bye!"

def extract_symptoms_self_report(info_dict):
    global user_symptoms
    entities_list = info_dict["entities"]
    global PersonSuffering
    suffering = True
    flag = False
    for entity in entities_list:
        if(entity["entity"]=="PersonSuffering"):
            PersonSuffering = entity["value"]
        elif (entity["entity"] == "suffering"):
            suffering = True
        elif (entity["entity"] == "notSuffering"):
            suffering = False
        elif (entity["entity"] == "symptom"):
            flag = True
            user_symptoms[entity["value"]] = suffering
    return flag

def get_info_dict(user_reply):
    text_dict = {"text": user_reply}
    process = subprocess.run(['curl', 'localhost:5005/model/parse', '-d', json.dumps(text_dict)],
                            stdout=subprocess.PIPE,
                            universal_newlines=True, stderr=subprocess.PIPE, text=True)
    return eval(process.stdout)

def predict_questions():
    return ["hard to swallow", "acid reflux", "nausea", "bloating and fever"]


def extract_symptoms_conversation(q, info_dict):
    global user_symptoms
    entities_list = info_dict["entities"]
    suffering = True
    flag = True
    questions = q.split(" and ")
    response4questions = []
    for entity in entities_list:
        if (entity["entity"] == "suffering"):
            suffering = True
            response4questions.append(suffering)
        elif (entity["entity"] == "notSuffering"):
            suffering = False
            response4questions.append(suffering)
        elif (entity["entity"] == "symptom"):
            user_symptoms[entity["value"]] = suffering
            flag = False
    if(flag):
        for i in range(len(questions)):
            user_symptoms[questions[i]] = response4questions[i]

def ask_questions(questions):
    for q in questions:
        no = random.randint(0, len(question_asking_templates)-1)
        # print("no:", no)
        if(no<2):
            print(question_asking_templates[no].format(q))
        else:
            if(PersonSuffering == "user"):
                print(question_asking_templates[no].format("you", q))
            else:
                print(question_asking_templates[no].format("your " + PersonSuffering, q))
        user_reply = input(">>")
        info_dict = get_info_dict(user_reply)
        extract_symptoms_conversation(q, info_dict)
        

def interact():
    global user_symptoms
    questions = []
    while(True):
        user_reply = input(">>")
        info_dict = get_info_dict(user_reply)
        if(info_dict['intent']["name"]=='greet'):
            print(give_greetings_reply())
        elif(info_dict['intent']["name"]=='self_report'):
            if(extract_symptoms_self_report(info_dict)):
                questions = predict_questions()
                ask_questions(questions)
                print(user_symptoms)
            else:
                print("There is no symptom reported. Please Report atleast one symptom!!")
                continue
        elif(info_dict['intent']["name"]=='goodbye'):
            print(give_bye_reply())
            break


if __name__=="__main__":
    interact()