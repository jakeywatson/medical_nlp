--------------------------------------------------------------------------------
### AHLT LAB 2 SUBMISSION ###
DATE: 03/05/2020
AUTHOR: JAKE WATSON
--------------------------------------------------------------------------------

This directory contains two directories containing python files for the
RULE BASED and MACHINE LEARNING tasks, and a third directory evaluator output
and raw output for both.

Necessary Packages: NetworkX, NLTK, Megam, Stanford CoreNLP

Before either program is run, start the CoreNLP server.

To run The RULE BASED classifier:
  1. Run the main.py file in the RULE BASED directory.

To run The MACHINE LEARNING classifier:
  1. Open the main.py file in the RULE BASED directory.
  2. Under line 56, comment out the lines as you wish to run the
     feature generation, training or classification.

     if __name__ == '__main__':
         #generate_features()
         #trainMegamMaxEnt()
         #trainNLTKMegam()
         main()

--------------------------------------------------------------------------------
