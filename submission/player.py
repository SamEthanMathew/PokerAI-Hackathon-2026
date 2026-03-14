from agents.libratus_agent import LibratusAgent


class PlayerAgent(LibratusAgent):
    """
    Tournament submission: Libratus-style agent with precomputed blueprint
    strategy (MCCFR) and equity-based discard.
    """

    def __name__(self):
        return "PlayerAgent"
