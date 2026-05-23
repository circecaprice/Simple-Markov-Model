import java.util.*;
import java.util.regex.*;
import java.io.*;
import compsci201.Ignore;

/**
 * AUTHOR: Eric Cai
 */

public class ClassifyingModel extends BaseMarkovModel{

    // declare more needed instance variables here
    private HashSet<String> vocabulary;
    private HashMap<String, Map<String, Integer>> occurences;
    private Map<String, List<String>> myMap;
    private Map<String, Map<String,Integer>> myCache;

    public ClassifyingModel(int size) {
        super(size);
        myMap = new HashMap<>();
        vocabulary = new HashSet<>();
        occurences = new HashMap<>();
        myCache = new HashMap<>();
            
        };
    

    public ClassifyingModel(){
        this(2);
    }

    /**
     * Returns the number of times token follows context in this model
     * @param context is an N-gram, a list of myOrder strings
     * @param token possibly follows the context in trained model
     * @return # occurrences of token following context in trained model
     */
    private int tokenInContextCount(List<String> context, String token) {
        
    String contextKey = String.join("\0", context);

    
    Map<String, Integer> theCache = myCache.get(contextKey);
    if (theCache != null && theCache.containsKey(token)) {
        return theCache.get(token);
    }

    
    if (theCache == null) {
        theCache = new HashMap<>();
        myCache.put(contextKey, theCache);
    }

   
    int count = 0;
    Map<String, Integer> freqMap = occurences.get(contextKey);
    if (freqMap != null) {
        count = freqMap.getOrDefault(token, 0);
    }

    
    theCache.put(token, count);

    return count;

    }

    /**
     * Use regular expression to tokenize rather
     * than split. Any alphabetic sequence followed by 
     * punctuation. So separation can be whitespace or number
     * for example.
     * @return list of tokens
     */

    @Ignore
    @Override
    public List<String> tokenize(String text){
        List<String> tokens = new ArrayList<>();
        String includePunc = "[A-Za-z]+|[.,!?;:]";
        Pattern pattern = Pattern.compile(includePunc);
        Matcher matcher = pattern.matcher(text.toLowerCase());

        while (matcher.find()) {
            tokens.add(matcher.group());
        }
        return tokens;
    }

    /**
     * Tokenize text and add <START>/<END> tags. Functionally
     * the same as BaseMarkovModel.updateWordSequence,
     * but that adds the padded text to an instance variable rather
     * than returning it. So code copied here
     * @param text to be processed
     * @return tokenized and pre/post padded sequence of tokens
     */
     private List<String> createTokenizedText(String text) {
        List<String> padded = new ArrayList<>();
        List<String> tokens = tokenize(text);

        for(int k=0; k < myModelSize; k++){
            padded.add("<START>");
        }
        padded.addAll(tokens);
        for(int k=0; k < myModelSize; k++){
            padded.add(END);
        }

        return padded;
    }

    /**
     * Return the log likelihood that text matches this trained model.
     * The text will be tokenized, then each n-gram/follow in text
     * "compared" in probabilistic way to the trained model's data.
     * Return the log probability of a match based on this comparison.
     * @param text is to be tokenized and matched against this model
     * @param smoother value used for Laplace smoothing
     * @return the normalized log probability of a match
     */
    public double calculateMatchProbability(String text, double smoother){

        List<String> padded = createTokenizedText(text);
    
    double probTotal = 0.0;
    for(int k=0; k < padded.size() - myModelSize; k++) {  

        List<String> context = new ArrayList<>(padded.subList(k, k + myModelSize));
        String contextKey = String.join("\0", context); // immutable key for lookup

        String token = padded.get(k + myModelSize);

        int contextCount = 0;
        if (myMap.containsKey(contextKey)) {
            contextCount = myMap.get(contextKey).size();
        }

        int nextCount = tokenInContextCount(context, token);

        double prob = (nextCount + smoother)/(contextCount + smoother * vocabulary.size()); 
        probTotal += Math.log(prob);
        
    }
    
    probTotal = probTotal / (padded.size() - myModelSize);
    return probTotal; // must be normalized before returning
    }

    @Override
    public void processTraining(){
        vocabulary.clear();
        myMap.clear();
        occurences.clear();
        // modify vocabulary instance variable

    for (int k = 0; k < myWordSequence.size() - myModelSize; k++) {
        
        List<String> current = new ArrayList<>();
        for (int j = 0; j < myModelSize; j++) {
            current.add(myWordSequence.get(k + j));
        }
        String contextKey = String.join("\0", current); // immutable key for context

        String next = myWordSequence.get(k + myModelSize);
        vocabulary.add(next);

        Map<String, Integer> frequency = occurences.get(contextKey);
        if (frequency == null) {
            frequency = new HashMap<>();
            occurences.put(contextKey, frequency);
        }
        frequency.put(next, frequency.getOrDefault(next, 0) + 1);

        myMap.putIfAbsent(contextKey, new ArrayList<>());
        myMap.get(contextKey).add(next);
    }
        }
    

    public int vocabularySize(){
        return vocabulary.size();
    }

    public static void main(String[] args) throws IOException {
        ClassifyingModel mm = new ClassifyingModel(3);
        String dirName = "data/shakespeare";
        mm.trainDirectory(dirName);
        System.out.printf("trained model for %s, vocab size = %d, tokens = %d\n",
                          dirName,mm.vocabularySize(),mm.tokenSize());
    }
}