import nltk
import re
import time
stopwords = nltk.download('stopwords')

stop_words = set(stopwords.words('english'))


def nlp_preprocessing(total_text, data_text, index, column):
    if type(total_text) is not int:
        string = ""
        # replace every special char with space
        total_text = re.sub('[^a-zA-Z0-9\n]', ' ', total_text)
        # replace multiple spaces with single space
        total_text = re.sub('\s+', ' ', total_text)
        # converting all the chars into lower-case.
        total_text = total_text.lower()

        for word in total_text.split():
            # if the word is a not a stop word then retain that word from the data
            if not word in stop_words:
                string += word + " "

        data_text[column][index] = string

        return data_text
    else:
        raise TypeError("found total text as a int.")

def nlp_runner(data_text):
    start_time = time.clock()
    result_text = data_text
    for index, row in data_text.iterrows():
        if type(row['TEXT']) is str:
            result_text = nlp_preprocessing(row['TEXT'], result_text,index, 'TEXT')
        else:
            print("there is no text description for id:", index)
    print('Time took for preprocessing the text :', time.clock() - start_time, "seconds")
    return result_text

 # call nlp_runner