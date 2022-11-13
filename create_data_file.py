with open('adult.data','r') as data_f:
    with open('adult.csv','w') as csv_f:
        id = 0
        csv_f.write("age, workclass, fnlwgt, education, education-num, marital-status, occupation, relationship, race, sex, capital-gain, capital-loss, hours-per-week, native-country")
