1. set up env
	conda env create -f environment.yml
2. show demo
	run sh ./demo.sh
	it will use examples.csv as input file and output a file output.csv, which contains predicted result
	*Note:	the seperator in all the csv files is '\t' (a tab character)
	*		each of the sequence is indentitfied by 'Entry name'.
	*		the file used to feed the model for prediction should contain 'Sequence' field

3. model generate
	enzyme_protein_classifier_model_generator.py is used to generate model for distinguish enzyme and non-enzyme. 
	This process will generate two files in a user specified folder.
	
	enzyme_classifier_model_generator.py is used to generate model for enzyme commision number prediction. 
	This process will generate two files in a user specified folder.

4. use model for prediction
	enzyme_protein_classifier.py use the model generated in step 3 to predict protein sequences contained in a CSV whether they are enzyme or non-enzyme.
	the file used for prediction should contain two fields, namely 'Sequence' and 'Entry name'. The CSV should use '\t' as seperator.
	
	enzyme_classifier.py use the model generated in step 3 to predict enzyme commision numbers of enzyme sequence. 
	The file used the file used for prediction should contain two fields, namely 'Sequence' and 'Entry name'. The CSV should use '\t' as seperator.
	
	*Note: when specify the model name to these two .py, the model name should not contain suffix. For example,
	*		a model generated in step 3 contains two file with same name but different suffix like enzyme_model.h5, enzyme_model.pkl
	*		when specifying them to .py described in this step, only "enzyme_model" should be specified.
	*
	