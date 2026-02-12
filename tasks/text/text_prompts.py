def get_text_prompt(text_prompt_id: str) -> str:
    return TEXT_PROMPTS[text_prompt_id]


TEXT_PROMPTS = {
    "text_prompt_1": "Draw a red circle in the center of the canvas",
    "text_prompt_2": "Draw a house with a triangular roof and rectangular body",
    "text_prompt_3": "Draw 4 squares in each corner of the canvas",
    "text_prompt_4": "Draw a red triangle inside a blue square",
    "text_prompt_5": "Draw a checkerboard pattern with 8x8 squares",
    "text_prompt_6": "Draw a blue rectangle",
    "text_prompt_7": "Draw a grid of 6 circles",
    "text_prompt_8": "Draw a road that narrows as it goes toward the top of the canvas",
    "text_prompt_9": "Draw two cubes, one larger in the foreground and one smaller in the background",
    "text_prompt_10": "Draw a set of railroad tracks converging toward a single vanishing point",
    "text_prompt_11": "Draw a row of trees that get smaller as they recede into the distance",
    "text_prompt_12": "Draw a circle partially hidden behind a square",
    "text_prompt_13": "Draw three rectangles stacked so that each overlaps the one below it",
    "text_prompt_14": "Draw a triangle touching the top edge of a square without overlapping it",
    "text_prompt_15": "Draw a small circle centered inside a larger circle",
    "text_prompt_19": "Draw a cube using perspective lines",
    "text_prompt_20": "Draw a staircase going upward into the distance",
    "text_prompt_21": "Draw a box where the front face is square and the side face is visible",
    "text_prompt_26": "Draw two identical squares, one appearing farther away than the other",
    "text_prompt_27": "Draw a hallway with walls, floor, and ceiling receding into the distance",
    "text_prompt_28": "Draw a cube rotated so no face is directly facing the viewer",
    "text_prompt_long_1": "Draw a small town scene using perspective. Start with a road that begins at the bottom center and narrows toward the horizon. Place a row of houses on the left side of the road and a row of trees on the right. Each house should face the road and get smaller as it goes into the distance. Add windows and doors to the houses, keeping their orientation consistent. Ensure that nearer objects overlap farther ones and that all elements align to the same vanishing point.",
    "deepmind_logo": "Draw the Deepmind logo in a hollow manner. This means no fill, only outlines.",
    "openai_logo": "Draw the OpenAI logo in a hollow manner. This means no fill, only outlines.",
    "claude_logo": "Draw the Claude logo in a hollow manner. This means no fill, only outlines.",
}
