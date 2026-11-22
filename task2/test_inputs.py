import unittest

from inputs import _normalize


class ExplicitConclusions(unittest.TestCase):
    def test_integer_conclusions(self):
        for text, want in (
            ("The integer is 007.", "7"),
            ("I first tried 14. Thus, the answer is the integer −12.", "-12"),
            ("An intermediate value was 6.\nThe printed value is 31.", "31"),
            ("Therefore, the output is 0.", "0"),
        ):
            with self.subTest(text=text):
                self.assertEqual(_normalize(text, "integer"), want)

    def test_question_specific_quantity(self):
        conclusion = "The minute field of the clock is 09."
        self.assertEqual(_normalize(conclusion, "integer", "What is the minute field?"), "9")
        self.assertIsNone(_normalize(conclusion, "integer", "What is the hour field?"))

    def test_explicit_choice(self):
        self.assertEqual(_normalize("I compared the options. Thus, the answer is C.", "choice"), "C")

    def test_rejects_ambiguous_or_incomplete_replies(self):
        for text, kind in (
            ("First calculate 19 plus 8.", "integer"),
            ("One possibility is x = 10.", "integer"),
            ("7 + 8 = 15", "integer"),
            ("The integer is 5 or 7.", "integer"),
            ("It would work if the answer is 4.", "integer"),
            ("Estimate: 3. The output is", "integer"),
            ("I obtained 10.\n11", "integer"),
            ("The result might be A. It could also be B.", "choice"),
            ("Therefore, the answer is A or C.", "choice"),
            ("<think>The answer is the integer 3.", "integer"),
        ):
            with self.subTest(text=text):
                self.assertIsNone(_normalize(text, kind))

    def test_plain_forms(self):
        self.assertEqual(_normalize("**Final answer:** **D**", "choice"), "D")
        self.assertEqual(_normalize("−002.0", "integer"), "-2")


if __name__ == "__main__":
    unittest.main()
