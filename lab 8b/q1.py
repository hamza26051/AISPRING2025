# Total number of cards in a deck
total_cards = 52

# Suits and Ranks
suits = ['hearts', 'diamonds', 'clubs', 'spades']
ranks = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A']

# Build the full deck
deck = [(rank, suit) for suit in suits for rank in ranks]

# 1. Probability of drawing a red card (hearts or diamonds)
red_cards = [card for card in deck if card[1] in ['hearts', 'diamonds']]
prob_red_card = len(red_cards) / total_cards

# 2. Given red card, probability it's a heart
hearts = [card for card in red_cards if card[1] == 'hearts']
prob_heart_given_red = len(hearts) / len(red_cards)

# 3. Given face card, probability it's a diamond
face_cards = [card for card in deck if card[0] in ['J', 'Q', 'K']]
diamond_faces = [card for card in face_cards if card[1] == 'diamonds']
prob_diamond_given_face = len(diamond_faces) / len(face_cards)

# 4. Given face card, probability it's a spade or a queen
spade_faces = [card for card in face_cards if card[1] == 'spades']
queen_faces = [card for card in face_cards if card[0] == 'Q']
# Use set to avoid double counting Q♠
spade_or_queen = set(spade_faces + queen_faces)
prob_spade_or_queen_given_face = len(spade_or_queen) / len(face_cards)

# Print results
print(f"1. Probability of red card: {prob_red_card:.2f}")
print(f"2. Probability of heart given red: {prob_heart_given_red:.2f}")
print(f"3. Probability of diamond given face card: {prob_diamond_given_face:.2f}")
print(f"4. Probability of spade or queen given face card: {prob_spade_or_queen_given_face:.2f}")
