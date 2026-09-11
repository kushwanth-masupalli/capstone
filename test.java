public class test {

    public static void main(String[] args) {

        String s = "226";

        int[] cost = {
            1,2,1,2,1,2,1,2,1,2,1,2,1,
            2,1,2,1,2,1,2,1,2,1,2,1,2
        };

        System.out.println(cost.length);

        System.out.println(findCost(s, cost));
    }

    public static int findCost(String s, int[] cost) {
        return helper(s, cost, 0, 0);
    }

    public static int helper(String s, int[] cost, int i, int ccost) {

        // IMPORTANT: i belongs to s, not cost
        if (i == s.length()) {
            return ccost;
        }

        int one = Integer.MAX_VALUE;
        int two = Integer.MAX_VALUE;

        // -----------------
        // Take ONE digit
        // -----------------

        if (s.charAt(i) != '0') {

            int num = s.charAt(i) - '0';

            one = helper(
                s,
                cost,
                i + 1,
                ccost + cost[num - 1]
            );
        }

        // -----------------
        // Take TWO digits
        // -----------------

        if (i + 1 < s.length()) {

            int num = Integer.parseInt(
                s.substring(i, i + 2)
            );

            if (num >= 10 && num <= 26) {

                two = helper(
                    s,
                    cost,
                    i + 2,
                    ccost + cost[num - 1]
                );
            }
        }

        return Math.min(one, two);
    }
}