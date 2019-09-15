/**
 * This class uses 2 methods - K-Nearest Neighbors (KNN) and RBF Network - to
 * solve 2 regression problems and used 3 methods - KNN, Condensed KNN, and RBF
 * Network to solve 2 classification problems. Results for all problems are 
 * written to an output file
 * 
 * @author Winston Lin
 */
import java.io.*;
import java.util.*;

public class WriteToFile 
{
    public static void main(String[] args) throws IOException
    {
        // 4 datasets used in this project
        String[] datasets = {"forestfires.data", "machine.data", "ecoli.data", 
                "segmentation.data"};
        // Write to the output file
        PrintWriter fout = new PrintWriter(new BufferedWriter(new FileWriter(
                "Results.txt", true)));
        fout.println("Perform regression and classification tasks. For "
                + "regression, test datasets with K-Nearest Neighbors and "
                + "RBF Network. Performance is measured by MSE. For "
                + "classification, test datasets with K-Nearest Neighbors, "
                + "Condensed K-Nearest Neighbors, and RBF Network. "
                + "Performance is measured by classification accuracy.");
        fout.println();
        fout.println(" ----------");
        fout.println("|REGRESSION|");
        fout.println(" ----------");
        fout.println();
        for (int i = 0; i < datasets.length; i++)
        {
            if (i < 2) // Regression
            {
                // Preprocess and normalize the dataset
                ProcessData data = new ProcessData(datasets[i]);
                data.process(true);
                data.kFold(5, false, "R");   
                
                // K-Nearest Neighbors
                KNN knn = new KNN("R");
                knn.test(data, false);
                
                // RBF Network
                RBFNetwork rbf = new RBFNetwork("R");
                double error = rbf.test(data, 0.001, 0.0001);
                
                // Write to the output file
                fout.println(data.fileName);
                fout.println("-----------");
                fout.println("K-Nearest Neighbors:");
                fout.println("Best k: " + knn.bestK);
                fout.println("Average MSE: " + knn.bestError);
                fout.println(); 
                fout.println("RBF Network:");
                fout.println("Average MSE: " + error);
                fout.println();
                if (i == 1)
                {
                    fout.println(" --------------");
                    fout.println("|CLASSIFICATION|");
                    fout.println(" --------------");
                    fout.println();
                }
            }
            else // Classification
            {
                // Preprocess and normalize the dataset
                ProcessData data = new ProcessData(datasets[i]);
                data.process(true);
                data.kFold(5, true, "C");   
                
                // K-Nearest Neighbors
                KNN knn = new KNN("C");
                knn.test(data, false);
                
                // Condensed K-Nearest Neighbors
                KNN cknn = new KNN("C");
                cknn.test(data, true);
                
                // RBF Network
                RBFNetwork rbf = new RBFNetwork("C");
                double accuracy = 0;
                if (i == 2)
                {// For ecoli.data
                    accuracy = rbf.test(data, 0.01, 0.75);
                }
                else
                {// For segmentation.data
                    accuracy = rbf.test(data, 100, 0.5);
                }
                
                double unnormalizedAccuracy = 0;
                if (i == 3)
                {// Additional test on segmentation.data without normalization
                    ProcessData unnormalizedData = new ProcessData(
                            "segmentation.data");
                    unnormalizedData.process(false); // Set normalize to false
                    unnormalizedData.kFold(5, true, "C");
                    RBFNetwork unnormalizedRbf = new RBFNetwork("C");
                    unnormalizedAccuracy = unnormalizedRbf.test(
                            unnormalizedData, 100, 0.65);
                }
                
                // Write to the output file
                fout.println(data.fileName);
                fout.println("-----------");
                fout.println("K-Nearest Neighbors:");
                fout.println("Best k: " + knn.bestK);
                fout.println("Average classification accuracy: " 
                        + knn.bestAccuracy);
                fout.println();    
                fout.println("Condensed K-Nearest Neighbors:");
                fout.println("Best k: " + cknn.bestK);
                fout.println("Average classification accuracy: " 
                        + cknn.bestAccuracy);
                fout.println(); 
                fout.println("RBF Network:");
                if (i == 2)
                {
                    fout.println("Average classification accuracy: " 
                            + accuracy);
                }
                if (i == 3)
                {
                    fout.println("Average classification accuracy (dataset "
                            + "normalized): " + accuracy);
                    fout.println("Average classification accuracy (dataset "
                            + "unnormalized): " + unnormalizedAccuracy);
                }
                fout.println();
            }
        }
        fout.close();
    }
}
