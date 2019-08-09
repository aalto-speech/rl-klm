__author__ = "Ulpu Remes"
__copyright__ = "Copyright (c) 2018, University of Helsinki"

# Generates unique combinations.

import numpy as np
import copy

def add_item(row,num_items,vocab_size,max_vocab_size):
    variations=[]
    for name in range(vocab_size):
        if name not in row:
            vari=copy.copy(row)
            vari.append(name)
            if vocab_size<max_vocab_size and name==vocab_size-1:
                vocab_size=vocab_size+1
            if len(vari)<num_items:
                variations=variations+add_item(vari,num_items,vocab_size,max_vocab_size)
            else:
                variations.append(vari)
    return variations
        
def namemap(index):
    return index+1

def find_buttons(transitions,printa=False):
    max_vocab_size=len(transitions)
    variations=[[]]
    for row in transitions:
        num_items=np.sum(row)
        variations_new=[]
        for vari in variations:
            vocab_size=min(max_vocab_size,len(np.unique(np.array(vari)))+1)
            new=add_item([],num_items,vocab_size,max_vocab_size)
            for newrow in new:
                variations_new.append(vari+[newrow])
        variations=copy.copy(variations_new)
    variations=[]
    for vari in variations_new:
        buttons=copy.deepcopy(transitions)
        for ii in range(len(buttons)):
            items=copy.copy(vari[ii])
            for jj in range(len(buttons[ii])):
                if buttons[ii][jj]>0:
                    buttons[ii][jj]=namemap(items.pop(0))
        variations.append(copy.deepcopy(buttons))

    # Transforming into correct form
    variations_dform = []
    for vari in variations:
        buttons_for_matrix = []
        for state in range(max_vocab_size):
            buttons_for_state = []
            for i in range(max_vocab_size):
                if vari[state][i] >= 1.:
                    buttons_for_state.append(vari[state][i]-1)
            buttons_for_matrix.append(buttons_for_state)
        variations_dform.append(buttons_for_matrix)
    if printa:
        for vari in variations:
            print(np.array(vari))
            print('')
    return variations_dform



