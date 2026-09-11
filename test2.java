import java.util.Arrays;

public class test {

    public static void main(String[] args) {

        int[] arr = {1, 2, 1, 3, 4, 5};

        int[] dp = new int[arr.length];
        Arrays.fill(dp, -1);

        System.out.println(helper(arr, 0, dp));
    }

    public static int helper(int[] arr, int i, int[] dp) {

        // Reached the last index
        if (i == arr.length - 1) {
            return 0;
        }

        // Already calculated
        if (dp[i] != -1) {
            return dp[i];
        }

        int one = Integer.MAX_VALUE;

        for (int j = i + 1;
             j <= Math.min(arr.length - 1, i + arr[i]);
             j++) {

            int next = helper(arr, j, dp);

            if (next != Integer.MAX_VALUE) {
                one = Math.min(one, next + 1);
            }
        }

        dp[i] = one;

        return dp[i];
    }
}