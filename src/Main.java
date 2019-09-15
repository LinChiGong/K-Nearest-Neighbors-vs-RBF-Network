/**
 * This class runs the whole project for demonstration purpose. It would accept 
 * an input file for regression and an input file for classification. Files are
 * processed and tested with KNN, Condensed KNN, and RBF Network. Running 
 * processes and results are printed to the console
 * 
 * @author Winston Lin
 */
import java.io.*;
import java.util.*;

public class Main 
{
    public static void main(String[] args) throws IOException
    {
        String regressionFile;
        String classificationFile;
        
        // Prompt for the dataset to be used
        Scanner scan = new Scanner(System.in);
        System.out.println();
        System.out.print("Enter input file name for regression: ");
        regressionFile = scan.nextLine().trim();
        System.out.println();
        
        // Preprocess and normalize the dataset for regression
        ProcessData data = new ProcessData(regressionFile);
        data.process(true);
        data.kFold(5, false, "R");
        System.out.println("Dataset has been processed and normalized");
        System.out.println();
        
        // For demonstration purpose
        System.out.print("Press 'Enter' to start testing with K-Nearest"
                + " Neighbors:");
        String temp = scan.nextLine().trim();
        System.out.println();
        
        // K-Nearest Neighbors for regression
        System.out.println("K-Nearest Neighbors");
        System.out.println("-------------------");
        KNN knn = new KNN("R");
        knn.test(data, false);
        System.out.println("Best k: " + knn.bestK);
        System.out.println("Best performance (minimum MSE): " + knn.bestError);
        System.out.println();
        
        // For demonstration purpose
        System.out.print("Press 'Enter' to start testing with RBF Network:");
        temp = scan.nextLine().trim();
        System.out.println();
        
        // RBF Network for regression
        System.out.println("RBF Network");
        System.out.println("-----------");
        RBFNetwork rbf = new RBFNetwork("R");
        // Set learning rate = 0.001 and convergence threshold = 0.0001
        double error = rbf.test(data, 0.001, 0.0001);
        
        // Prompt for the dataset to be used
        System.out.println();
        System.out.print("Enter input file name for classification: ");
        classificationFile = scan.nextLine().trim();
        System.out.println();
        
        // Preprocess and normalize the dataset for classification
        data = new ProcessData(classificationFile);
        data.process(true);
        data.kFold(5, true, "C");
        System.out.println("Dataset has been processed and normalized");
        System.out.println();
        
        // For demonstration purpose
        System.out.print("Press 'Enter' to start testing with K-Nearest"
                + " Neighbors:");
        temp = scan.nextLine().trim();
        System.out.println();
        
        // K-Nearest Neighbors for classification
        System.out.println("K-Nearest Neighbors");
        System.out.println("-------------------");
        knn = new KNN("C");
        knn.test(data, false);
        System.out.println("Best k: " + knn.bestK);
        System.out.println("Best performance (maximum accuracy): " 
                + knn.bestAccuracy);
        System.out.println();
        
        // For demonstration purpose
        System.out.print("Press 'Enter' to start testing with Condensed "
                + "K-Nearest Neighbors:");
        temp = scan.nextLine().trim();
        System.out.println();
        
        // Condensed K-Nearest Neighbors for classification
        System.out.println("Condensed K-Nearest Neighbors");
        System.out.println("-----------------------------");
        knn = new KNN("C");
        knn.test(data, true);
        System.out.println("Best k: " + knn.bestK);
        System.out.println("Best performance (maximum accuracy): " 
                + knn.bestAccuracy);
        System.out.println();
        
        // For demonstration purpose
        System.out.print("Press 'Enter' to start testing with RBF Network:");
        temp = scan.nextLine().trim();
        System.out.println();
        
        // RBF Network for regression
        System.out.println("RBF Network");
        System.out.println("-----------");
        rbf = new RBFNetwork("C");
        // Set learning rate = 0.01 and convergence threshold = 0.75
        double accuracy = rbf.test(data, 0.01, 0.75);
        System.out.println();

        scan.close();
    }
}
