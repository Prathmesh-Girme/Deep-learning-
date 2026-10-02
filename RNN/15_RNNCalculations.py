# ht = tanh(Wx * Xt + Wh * ht-1 + b)

#Xt            current input
#Wt            weight of current input
#Wh            weight of previous hidden state
#b             bias
#ht-1          previous hidden state
# tanh         Activation function(-1 to 1)
# ht           New hidden state


import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))








def MarvellousRNNPredictions():
    print("Calculations of RNN")
    # food was not good
    inputs = [1,2,5,3]
    
    hidden_state = 0
    
    #RNN Parameters
    Wx = 0.5
    Wh = 0.8
    b = 0.1
    
    #11:46
    
    #RNN Calculation
    
    for time_step, X in enumerate(inputs):
        previous_hidden_state = hidden_state
        
        weighted_input = Wx * X
        weighted_memory = Wh * previous_hidden_state
        
        total = weighted_input + weighted_memory + b
        hidden_state = np.tanh(total)
        
        print("Timestep:", time_step+1)
        print("Input : " , X)
        print("Hidden State : ", hidden_state)
        print("-"*30)
    #step 2 :Final hidden state   
    print("Final Hidden state : ", hidden_state)
    
    # step 3 :Output Layer
    #Output = Wy * FinalHiddenState + Output Bias
    
    Wy = 1.0 
    output_bias = 0.0
    
    output = (Wy * hidden_state) + output_bias
    
    print("Raw Output :" , output)
    
    

    
    

def main():
    MarvellousRNNPredictions()
    
    
    
    
if __name__ == "__main__":
    main()