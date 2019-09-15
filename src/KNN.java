/**
 * This class implements the K-Nearest Neighbors Algorithm. It handles both 
 * regression and classification tasks. The test() method takes in datasets 
 * that are already preprocessed by the ProcessData class. The inner class
 * Neighbor stores the information of a neighbor of the query point. This class
 * also implements the Condensed K-Nearest Neighbors
 * 
 * @author Winston Lin
 */
import java.io.*;
import java.util.*;

public class KNN 
{
    class Neighbor implements Comparable<Neighbor>
    {   
        // Information of the neighbor
        double[] X;
        double yValue;
        String yLabel;
        double distance = 0;
        
        public Neighbor(double[] X, double yValue)
        {// Initialization for regression
            this.X = X;
            this.yValue = yValue;
        }
        
        public Neighbor(double[] X, String yLabel)
        {// Initialization for classification
            this.X = X;
            this.yLabel = yLabel;
        }

        /**
         * This method allows sorting based on distance        
         */
        public int compareTo(Neighbor neighbor)
        {
            return Double.compare(this.distance, neighbor.distance);
        }
    }
    
    String task = ""; // Type of the task
    ArrayList<ArrayList<double[]>> XPartitions = 
            new ArrayList<ArrayList<double[]>>();
    ArrayList<double[]> XTrain = new ArrayList<double[]>();
    ArrayList<double[]> XTest = new ArrayList<double[]>();
    // For regression tasks
    ArrayList<ArrayList<Double>> yValuePartitions = 
            new ArrayList<ArrayList<Double>>();
    ArrayList<Double> yValueTrain = new ArrayList<Double>();
    ArrayList<Double> yValueTest = new ArrayList<Double>();
    // For classification tasks
    ArrayList<ArrayList<String>> yLabelPartitions = 
            new ArrayList<ArrayList<String>>();
    ArrayList<String> yLabelTrain = new ArrayList<String>();
    ArrayList<String> yLabelTest = new ArrayList<String>();  
    // For Condensed KNN
    ArrayList<double[]> condensedXTrain = new ArrayList<double[]>();
    ArrayList<String> condensedYLabelTrain = new ArrayList<String>();
    
    int bestK = 0; // Number of neighbors (k) with the best performance
    double bestError = 0;    // Minimum MSE
    double bestAccuracy = 0; // Maximum classification accuracy
    ArrayList<Double> errors = new ArrayList<Double>(); // Error for each fold
    ArrayList<Double> accuracies = new ArrayList<Double>(); // Accuracy / fold
    
    public KNN(String task)
    {// "R" for regression task and "C" for classification task
        this.task = task;
    }
    
    /**
     * This method handles regression and classification tasks separately
     * 
     * @param data is the k-fold partitioned data processed by ProcessData
     * @param condensed suggests using Condensed KNN or not
     */
    public void test(ProcessData data, boolean condensed)
    {
        this.XPartitions = data.XPartitions;
        this.yValuePartitions = data.yValuePartitions;
        this.yLabelPartitions = data.yLabelPartitions;
        
        int k = 3; // Start with 3 neighbors
        if (task.equalsIgnoreCase("R"))
        {// Select best k for regression task
            double minError = Double.MAX_VALUE;
            boolean hasImproved = true;
            while (hasImproved)
            {
                double error = KNNRegressor(k);
                if (error < minError)
                {
                    minError = error;
                    k++;
                }
                else
                {
                    k--;
                    hasImproved = false;
                }
            }
            bestError = minError;
        }
        else if (task.equalsIgnoreCase("C") && !condensed)
        {// Select best k for classification task with regular KNN
            double maxAccuracy = Double.MIN_VALUE;
            boolean hasImproved = true;
            while (hasImproved)
            {
                double accuracy = KNNClassifier(k);
                if (accuracy > maxAccuracy)
                {
                    maxAccuracy = accuracy;
                    k++;
                }
                else
                {
                    k--;
                    hasImproved = false;
                }
            }
            bestAccuracy = maxAccuracy;
        }
        else
        {// Select best k for classification task with Condensed KNN
            double maxAccuracy = Double.MIN_VALUE;
            boolean hasImproved = true;
            while (hasImproved)
            {
                double accuracy = condensedKNNClassifier(k);
                if (accuracy > maxAccuracy)
                {
                    maxAccuracy = accuracy;
                    k++;
                }
                else
                {
                    k--;
                    hasImproved = false;
                }
            }
            bestAccuracy = maxAccuracy;
        }
        bestK = k;
    }
    
    /**
     * This method handles regression tasks
     * 
     * @param k is the number of neighbors
     * @return average MSE from cross validation
     */
    public double KNNRegressor(int k)
    {
        for (int i = 0; i < XPartitions.size(); i++)
        {// Perform KNN with each train-test pair
            XTest = XPartitions.get(i);
            yValueTest = yValuePartitions.get(i);
            for (int j = 0; j < XPartitions.size(); j++)
            {
                if (j != i)
                {
                    for (double[] X : XPartitions.get(j))
                    {
                        XTrain.add(X);
                    }
                    for (double yValue : yValuePartitions.get(j))
                    {
                        yValueTrain.add(yValue);
                    }
                }
            }
            ArrayList<Double> yPredicts = new ArrayList<Double>();
            for (int m = 0; m < XTest.size(); m++)
            {// For each test sample, make a prediction about its target value
                ArrayList<Neighbor> neighbors = new ArrayList<Neighbor>();
                for (int n = 0; n < XTrain.size(); n++)
                {// Get the distance to all instances in the training set
                    Neighbor neighbor = new Neighbor(XTrain.get(n),
                            yValueTrain.get(n));
                    neighbor.distance = getDistance(XTrain.get(n), 
                            XTest.get(m));
                    neighbors.add(neighbor);
                }
                Collections.sort(neighbors); // Sort neighbors by distance
                double yPredict = 0;
                for (int s = 0; s < k; s++)
                {// Prediction is the mean value of k neighbors
                    yPredict += neighbors.get(s).yValue;
                }
                yPredict /= k;       
                yPredicts.add(yPredict);
            }
            double error = 0;
            for (int t = 0; t < yPredicts.size(); t++)
            {// Calculate the mean squared error for this fold
                error += Math.pow((yPredicts.get(t) - yValueTest.get(t)), 2);
            }
            error /= yPredicts.size();
            errors.add(error);
            
            // For demonstration purpose
            System.out.println("Error: " + error);
            
            XTrain.clear();
            yValueTrain.clear();
        }
        // Calculate average error from all k folds
        double averageError = 0;
        for (Double error : errors)
        {
            averageError += error;
        }
        averageError /= XPartitions.size();
        errors.clear();
        
        // For demonstration purpose
        System.out.println("-------------------------------");
        System.out.println("k = " + k);
        System.out.println("Average error = " + averageError);
        System.out.println("-------------------------------");
        
        return averageError;
    }
    
    /**
     * This method calculates the Euclidean distance between two points
     * 
     * @param x1 is the first point
     * @param x2 is the second point
     * @return the Euclidean distance
     */
    public double getDistance(double[] x1, double[] x2)
    {
        double distance = 0;
        for (int i = 0; i < x1.length; i++)
        {
            distance += Math.pow((x1[i] - x2[i]), 2);
        }
        return Math.sqrt(distance);
    }
            
    /**
     * This method handles classification tasks using regular KNN
     * 
     * @param k is the number of neighbors
     * @return average classification accuracy from cross validation
     */
    public double KNNClassifier(int k)
    {
        for (int i = 0; i < XPartitions.size(); i++)
        {// Perform KNN with each train-test pair
            XTest = XPartitions.get(i);
            yLabelTest = yLabelPartitions.get(i);
            for (int j = 0; j < XPartitions.size(); j++)
            {
                if (j != i)
                {
                    for (double[] X : XPartitions.get(j))
                    {
                        XTrain.add(X);
                    }
                    for (String yLabel : yLabelPartitions.get(j))
                    {
                        yLabelTrain.add(yLabel);
                    }
                }
            }
            ArrayList<String> yPredicts = new ArrayList<String>();
            for (int m = 0; m < XTest.size(); m++)
            {// For each test sample, make a prediction about its label
                ArrayList<Neighbor> neighbors = new ArrayList<Neighbor>();
                for (int n = 0; n < XTrain.size(); n++)
                {// Get the distance to all instances in the training set
                    Neighbor neighbor = new Neighbor(XTrain.get(n), 
                            yLabelTrain.get(n));
                    neighbor.distance = getDistance(XTrain.get(n), 
                            XTest.get(m));
                    neighbors.add(neighbor);
                }
                Collections.sort(neighbors); // Sort neighbors by distance
                HashMap<String, Integer> labelCount = new HashMap<String, 
                        Integer>(); // Stores the count of each label
                String yPredict = "";
                for (int s = 0; s < k; s++)
                {// Count the labels of k neighbors
                    if (labelCount.containsKey(neighbors.get(s).yLabel))
                    {
                        int count = labelCount.get(neighbors.get(s).yLabel);
                        count++;
                        labelCount.put(neighbors.get(s).yLabel, count);
                    }
                    else
                    {
                        labelCount.put(neighbors.get(s).yLabel, 1);
                    }
                }
                Iterator<Map.Entry<String, Integer>> it = 
                        labelCount.entrySet().iterator();
                int max = Integer.MIN_VALUE;
                while(it.hasNext())
                {// Determine the label with highest count
                    Map.Entry<String, Integer> pair = it.next();
                    if (pair.getValue() > max)
                    {
                        max = pair.getValue();
                        yPredict = pair.getKey();
                    }
                }
                yPredicts.add(yPredict);
            }
            double correctPrediction = 0;
            for (int t = 0; t < yPredicts.size(); t++)
            {// Calculate the mean squared error for this fold
                if (yPredicts.get(t).equals(yLabelTest.get(t)))
                {
                    correctPrediction++;
                }
            }
            double accuracy = correctPrediction / yPredicts.size();
            accuracies.add(accuracy);
            
            // For demonstration purpose
            System.out.println("Accuracy: " + accuracy);
            
            XTrain.clear();
            yLabelTrain.clear();
        }
        // Calculate average accuracy from all k folds
        double averageAccuracy = 0;
        for (Double accuracy : accuracies)
        {
            averageAccuracy += accuracy;
        }
        averageAccuracy /= XPartitions.size();
        accuracies.clear();
        
        // For demonstration purpose
        System.out.println("-------------------------------");
        System.out.println("k = " + k);
        System.out.println("Average accuracy = " + averageAccuracy);
        System.out.println("-------------------------------");
        
        return averageAccuracy;
    }
    
    /**
     * This method handles classification tasks using Condensed KNN
     * 
     * @param k is the number of neighbors
     * @return average classification accuracy from cross validation
     */
    public double condensedKNNClassifier(int k)
    {
        for (int i = 0; i < XPartitions.size(); i++)
        {// Perform KNN with each train-test pair
            XTest = XPartitions.get(i);
            yLabelTest = yLabelPartitions.get(i);
            for (int j = 0; j < XPartitions.size(); j++)
            {
                if (j != i)
                {
                    for (double[] X : XPartitions.get(j))
                    {
                        XTrain.add(X);
                    }
                    for (String yLabel : yLabelPartitions.get(j))
                    {
                        yLabelTrain.add(yLabel);
                    }
                }
            }
            // Create a condensed subset
            ArrayList<Integer> unusedSample= new ArrayList<Integer>();
            for (int r = 0; r < XTrain.size(); r++)
            {
                unusedSample.add(r);
            }
            boolean isStable = false;
            while (!isStable && !unusedSample.isEmpty())
            {// Continue to add instances to the subset till it is stable 
                if (condensedXTrain.isEmpty())
                {// Add the first point
                    Random random = new Random();
                    int selectSample = random.nextInt(unusedSample.size());
                    condensedXTrain.add(XTrain.get(unusedSample.get(
                            selectSample)));
                    condensedYLabelTrain.add(yLabelTrain.get(unusedSample.get(
                            selectSample)));
                    unusedSample.remove(selectSample); // Remove visited
                }
                else
                {
                    Collections.shuffle(unusedSample);
                    isStable = true;
                    for (int p = 0; p < unusedSample.size(); p++)
                    {
                        ArrayList<Neighbor> neighbors = 
                                new ArrayList<Neighbor>();
                        for (int q = 0; q < condensedXTrain.size(); q++)
                        {// Get the distance to all instances in condensed set
                            Neighbor neighbor = new Neighbor(
                                    condensedXTrain.get(q), 
                                    condensedYLabelTrain.get(q));
                            neighbor.distance = getDistance(
                                    condensedXTrain.get(q), 
                                    XTrain.get(unusedSample.get(p)));
                            neighbors.add(neighbor);
                        }
                        Collections.sort(neighbors); // Sort by distance
                        // Compare training data point to its nearest neighbor
                        if (!neighbors.get(0).yLabel.equals(yLabelTrain.get(
                                unusedSample.get(p))))
                        {// Add instance to the condensed subset if not agree
                            condensedXTrain.add(XTrain.get(unusedSample.get(
                                    p)));
                            condensedYLabelTrain.add(yLabelTrain.get(
                                    unusedSample.get(p)));
                            unusedSample.remove(p);
                            p--; // Account for the removed instance
                            isStable = false;
                        }
                    }
                }
            }
            ArrayList<String> yPredicts = new ArrayList<String>();
            for (int m = 0; m < XTest.size(); m++)
            {// For each test sample, make a prediction about its label
                ArrayList<Neighbor> neighbors = new ArrayList<Neighbor>();
                for (int n = 0; n < condensedXTrain.size(); n++)
                {// Get the distance to all instances in the condensed subset
                    Neighbor neighbor = new Neighbor(condensedXTrain.get(n), 
                            condensedYLabelTrain.get(n));
                    neighbor.distance = getDistance(condensedXTrain.get(n), 
                            XTest.get(m));
                    neighbors.add(neighbor);
                }
                Collections.sort(neighbors); // Sort neighbors by distance
                HashMap<String, Integer> labelCount = new HashMap<String, 
                        Integer>(); // Stores the count of each label
                String yPredict = "";
                for (int s = 0; s < k; s++)
                {// Count the labels of k neighbors
                    if (labelCount.containsKey(neighbors.get(s).yLabel))
                    {
                        int count = labelCount.get(neighbors.get(s).yLabel);
                        count++;
                        labelCount.put(neighbors.get(s).yLabel, count);
                    }
                    else
                    {
                        labelCount.put(neighbors.get(s).yLabel, 1);
                    }                   
                }
                Iterator<Map.Entry<String, Integer>> it = 
                        labelCount.entrySet().iterator();
                int max = Integer.MIN_VALUE;
                while(it.hasNext())
                {// Determine the label with highest count
                    Map.Entry<String, Integer> pair = it.next();
                    if (pair.getValue() > max)
                    {
                        max = pair.getValue();
                        yPredict = pair.getKey();
                    }
                }
                yPredicts.add(yPredict);
            }
            double correctPrediction = 0;
            for (int t = 0; t < yPredicts.size(); t++)
            {// Calculate the mean squared error for this fold
                if (yPredicts.get(t).equals(yLabelTest.get(t)))
                {
                    correctPrediction++;
                }
            }
            double accuracy = correctPrediction / yPredicts.size();
            accuracies.add(accuracy);
            
            // For demonstration purpose
            System.out.print("Training set size: " + XTrain.size() + " --> ");
            System.out.println(condensedXTrain.size());
            
            XTrain.clear();
            yLabelTrain.clear();
            condensedXTrain.clear();
            condensedYLabelTrain.clear();
        }
        // Calculate average accuracy from all k folds
        double averageAccuracy = 0;
        for (Double accuracy : accuracies)
        {
            averageAccuracy += accuracy;
        }
        averageAccuracy /= XPartitions.size();
        accuracies.clear();
        
        // For demonstration purpose
        System.out.println("-------------------------------");
        System.out.println("k = " + k);
        System.out.println("Average accuracy = " + averageAccuracy);
        System.out.println("-------------------------------");
        
        return averageAccuracy;
    }
}
