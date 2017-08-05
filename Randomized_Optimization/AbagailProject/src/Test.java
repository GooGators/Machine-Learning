import dist.*;
import opt.*;
import opt.example.*;
import opt.ga.*;
import shared.*;
import func.nn.backprop.*;

import java.util.*;
import java.io.*;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.text.*;

/**
 * Implementation of randomized hill climbing, simulated annealing, and genetic algorithm to
 * find optimal weights to a neural network that is classifying abalone as having either fewer 
 * or more than 15 rings. 
 *
 * @author Hannah Lau
 * @version 1.0
 */
public class Test {
    private static Instance[] instances = initializeInstances();

    private static int inputLayer = 8, hiddenLayerOne = 6, hiddenLayerTwo = 4, outputLayer = 1, trainingIterations = 4000;
    private static BackPropagationNetworkFactory factory = new BackPropagationNetworkFactory();
    
    private static ErrorMeasure measure = new SumOfSquaresError();



    private static BackPropagationNetwork networks[] = new BackPropagationNetwork[3];
    private static NeuralNetworkOptimizationProblem[] nnop = new NeuralNetworkOptimizationProblem[3];

    private static OptimizationAlgorithm[] oa = new OptimizationAlgorithm[3];
    private static String[] oaNames = {"RHC", "SA", "GA"};
    private static String results = "";
    private static Instance[] train2 = Arrays.copyOfRange(instances, 0, 537);
    private static Instance[] test2 = Arrays.copyOfRange(instances, 538, 768);
    private static DataSet set = new DataSet(train2);
    private static DecimalFormat df = new DecimalFormat("00.000");
   // private static DecimalFormat df2 = new DecimalFormat(".000");

    public static void main(String[] args) {
        for(int i = 0; i < oa.length; i++) {
            networks[i] = factory.createClassificationNetwork(
                new int[] {inputLayer, hiddenLayerOne, hiddenLayerTwo,outputLayer});
            nnop[i] = new NeuralNetworkOptimizationProblem(set, networks[i], measure);
        }

        oa[0] = new RandomizedHillClimbing(nnop[0]);
        oa[1] = new SimulatedAnnealing(1E11, .95, nnop[1]);
        oa[2] = new StandardGeneticAlgorithm(100, 50, 10, nnop[2]);

        for(int i = 0; i < oa.length; i++) {
            double start = System.nanoTime(), end, trainingTime, testingTime, correct = 0, incorrect = 0;
            train(oa[i], networks[i], oaNames[i]); //trainer.train();
            end = System.nanoTime();
            trainingTime = end - start;
            trainingTime /= Math.pow(10,9);

            Instance optimalInstance = oa[i].getOptimal();
            networks[i].setWeights(optimalInstance.getData());

            double predicted, actual;
            start = System.nanoTime();
            for(int j = 0; j < train2.length; j++) {
                networks[i].setInputValues(train2[j].getData());
                networks[i].run();

                predicted = Double.parseDouble(train2[j].getLabel().toString());
                actual = Double.parseDouble(networks[i].getOutputValues().toString());

                double trash = Math.abs(predicted - actual) < 0.5 ? correct++ : incorrect++;

            }
            end = System.nanoTime();
            testingTime = end - start;
            testingTime /= Math.pow(10,9);



            //test
            start = System.nanoTime();
            correct = 0; incorrect = 0;
            for(int j = 0; j < test2.length; j++) {
                networks[i].setInputValues(test2[j].getData());
                networks[i].run();

                predicted = Double.parseDouble(test2[j].getLabel().toString());
                actual = Double.parseDouble(networks[i].getOutputValues().toString());

                double trash = Math.abs(predicted - actual) < 0.5 ? correct++ : incorrect++;
                
            }
         
            end = System.nanoTime();
            testingTime = end - start;
            testingTime /= Math.pow(10,9);

            results +=  "\nTest Results for " + oaNames[i] + ": \nCorrectly classified " + correct + " instances." +
                    "\nIncorrectly classified " + incorrect + " instances.\nPercent correctly classified: "
                    + df.format(correct/(correct+incorrect)*100) + "%\nTraining time: " + df.format(trainingTime)
                    + " seconds\nTesting time: " + df.format(testingTime) + " seconds\n";
            start = System.nanoTime();
         
        String training_accuracy = df.format(calculate_accuracy(train2, optimalInstance));
        String test_accuracy = df.format(calculate_accuracy(test2, optimalInstance));
        String training_time = df.format(trainingTime);

        results +=
                "\nTraining Accuracy: " + training_accuracy + "%\n"
                + "Testing Accuracy: " + test_accuracy + "%\n"
                + "Training time: " + training_time + " seconds";
        System.out.println("Test/Train");
        //System.out.println(results);
        //write_output_to_file("/Users/tyler/Desktop", "the.txt", results, false);
        //write_output_to_file("/Users/tyler/Desktop", "the.txt", "," + training_accuracy + "," + test_accuracy + "," + training_time, true);
            
        }

        System.out.println(results);
        //write_output_to_file("/Users/tyler/Desktop", "the.txt", results, true);
    }
    
    private static double calculate_accuracy(Instance[] instances, Instance optimalInstance) {
        int correct = 0, incorrect = 0;
        BackPropagationNetwork network = new BackPropagationNetworkFactory().createClassificationNetwork(new int[] {inputLayer, hiddenLayerOne, hiddenLayerTwo,outputLayer});
        network.setWeights(optimalInstance.getData());
        for(int j = 0; j < instances.length; j++) {
            network.setInputValues(instances[j].getData());
            network.run();

            double predicted = Double.parseDouble(instances[j].getLabel().toString());
            double actual = Double.parseDouble(network.getOutputValues().toString());

            double trash = Math.abs(predicted - actual) < 0.5 ? correct++ : incorrect++;
        }
        return correct*100.0/(correct+incorrect);
    }
    public static void write_output_to_file(String output_dir, String file_path, String result, Boolean final_result) {
    	// This function will have to be modified depending on the format of your file name.
    	// Else the splits won't work.
    	try {
    	if (final_result) {
    	String full_path = output_dir + "/" + new SimpleDateFormat("yyyy-MM-dd").format(new Date()) +
    	"/" + "final_result.csv";
    	String[] params = file_path.split("_");
    	String line = "";
    	switch (params.length) {
    	case 9:
    	line = params[0] + ",none," + params[6] + "," + params[8].split("\\.")[0];
    	break;
    	case 10:
    	line = params[0] + "," + params[3] + "," + params[7] + "," + params[9].split("\\.")[0];
    	break;
    	case 11:
    	line = params[0] + "," + params[3] + "_" + params[4] + "," + params[8] + ","
    	+ params[10].split("\\.")[0];
    	break;
    	}
    	PrintWriter pwtr = new PrintWriter(new BufferedWriter(new FileWriter(full_path, true)));
    	synchronized (pwtr) {
    	pwtr.println(line + result);
    	pwtr.close();
    	}
    	} else {
    	String full_path = output_dir + "/" + new SimpleDateFormat("yyyy-MM-dd").format(new Date()) +
    	"/" + file_path;
    	Path p = Paths.get(full_path);
    	Files.createDirectories(p.getParent());
    	Files.write(p, result.getBytes());
    	}
    	} catch (Exception e) {
    	e.printStackTrace();
    	}
    }
    private static void train(OptimizationAlgorithm oa, BackPropagationNetwork network, String oaName) {
        System.out.println("\nError results for " + oaName + "\n---------------------------");

        for(int i = 0; i < trainingIterations; i++) {
            oa.train();

            double train_error = 0;
            for(int j = 0; j < train2.length; j++) {
                network.setInputValues(train2[j].getData());
                network.run();

                Instance output = train2[j].getLabel(), example = new Instance(network.getOutputValues());
                example.setLabel(new Instance(Double.parseDouble(network.getOutputValues().toString())));
                train_error += measure.value(output, example);
            }

            double test_error = 0;
            for(int j = 0; j < test2.length; j++) {
                network.setInputValues(test2[j].getData());
                network.run();

                Instance output = test2[j].getLabel(), example = new Instance(network.getOutputValues());
                example.setLabel(new Instance(Double.parseDouble(network.getOutputValues().toString())));
                test_error += measure.value(output, example);
            }

           // System.out.println(df.format(train_error));
            System.out.println(df.format(test_error));
        }
    }

    private static Instance[] initializeInstances() {

        double[][][] attributes = new double[768][][];

        try {
            BufferedReader br = new BufferedReader(new FileReader(new File("/Users/tyler/Desktop/diabetes.txt")));

            for(int i = 0; i < attributes.length; i++) {
                Scanner scan = new Scanner(br.readLine());
                scan.useDelimiter(",");

                attributes[i] = new double[2][];
                attributes[i][0] = new double[8]; // 8 attributes
                attributes[i][1] = new double[1];

                for(int j = 0; j < 8; j++)
                    attributes[i][0][j] = Double.parseDouble(scan.next());

                attributes[i][1][0] = Double.parseDouble(scan.next());
            }
        }
        catch(Exception e) {
            e.printStackTrace();
        }

        Instance[] instances = new Instance[attributes.length];

        for(int i = 0; i < instances.length; i++) {
            instances[i] = new Instance(attributes[i][0]);
            // classifications range from 0 to 30; split into 0 - 14 and 15 - 30
            //[i].setLabel(new Instance(attributes[i][1][0] < 15 ? 0 : 1));
            //instances[i] = new Instance(attributes[i][0]); 
            instances[i].setLabel(new Instance(Math.floor(attributes[i][1][0])));
        }

        	
        return instances;
     
    }

}
