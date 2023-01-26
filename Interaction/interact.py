def give_greetings_reply():
    return "Hey! How can I help you?"

def give_bye_reply():
    return "Bye!"

def extract_symptoms(info_dict):
    print()

def get_info_dict(user_input):
    

def interact():
    while(True):
        user_input = input("Please enter the text:")
        info_dict = get_info_dict(user_input)
        if(info_dict['intent']=='greetings'):
            print(give_greetings_reply())
        elif(info_dict['intent']=='self_report'):
            extract_symptoms(info_dict)
        elif(info_dict['intent']=='good_bye'):
            print(give_bye_reply)