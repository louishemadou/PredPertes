from dataclasses import dataclass

@dataclass
class Bar:
    """Knowing estimated computation time (can be any unitless arbitrary value),
    update a loading bar on each call given current remaining time.
    """
    total: int
    message: str="Progress"
    advanced: int=0

    def advance(self, step):
        """Print simple loading bar, just knowing remaining 'time'."""
        self.advanced = min(self.total, self.advanced + step)
        ratio = self.advanced / self.total
        progress = round(50 * ratio)
        bar = "█" * progress + "-" * (50 - progress)
        print(f"{self.message:<10}: |{bar}| {ratio * 100:.1f}% Complete", end="\r")

    def __del__(self):
        self.advance(self.total)
        print()
