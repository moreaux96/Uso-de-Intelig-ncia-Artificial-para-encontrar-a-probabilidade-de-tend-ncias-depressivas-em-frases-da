#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# função para retornar o predict
# voce pode usar joblib ou pickel para importar modelos prontos

def depression(s):
    pred = model.predict([s])
    predprob = model.predict_proba([s])
    if pred[0] == 1:
        return print('Não depressivo - Probabilidade : ', np.max(predprob))
    else:
         return print('Depressivo - Probabilidade : ', np.max(predprob))
        
print(depression('Eu te amo'))
new_text = input('Digite seu texto aqui')
print(depression(new_text))
