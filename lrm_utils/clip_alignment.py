import torch.nn as nn
import clip
import torch
import numpy as np
from pdb import set_trace
from dalle2_pytorch import OpenAIClipAdapter, DiffusionPriorNetwork, DiffusionPrior

weight_urls = dict(
    alexnet_lrm1='https://github.com/harvard-visionlab/lrm-steering/releases/download/prototypes-v1/clip_diffusion_aligned_ep4_ViT_B16_to_alexnet_lrm1_40b29a3427-6016e8d4.pth',
    alexnet_lrm2='https://github.com/harvard-visionlab/lrm-steering/releases/download/prototypes-v1/clip_diffusion_aligned_ep4_ViT_B16_to_alexnet_lrm2_17b4229a30-59b5b992.pth',
    alexnet_lrm3='https://github.com/harvard-visionlab/lrm-steering/releases/download/prototypes-v1/clip_diffusion_aligned_ep4_ViT_B16_to_alexnet_lrm3_63ab1b3b06-749a3f15.pth',
)

class PromptEnsemble(nn.Module):
    
    imagenet_classes = ["tench", "goldfish", "great white shark", "tiger shark", "hammerhead shark", "electric ray", "stingray", "rooster", "hen", "ostrich", "brambling", "goldfinch", "house finch", "junco", "indigo bunting", "American robin", "bulbul", "jay", "magpie", "chickadee", "American dipper", "kite (bird of prey)", "bald eagle", "vulture", "great grey owl", "fire salamander", "smooth newt", "newt", "spotted salamander", "axolotl", "American bullfrog", "tree frog", "tailed frog", "loggerhead sea turtle", "leatherback sea turtle", "mud turtle", "terrapin", "box turtle", "banded gecko", "green iguana", "Carolina anole", "desert grassland whiptail lizard", "agama", "frilled-necked lizard", "alligator lizard", "Gila monster", "European green lizard", "chameleon", "Komodo dragon", "Nile crocodile", "American alligator", "triceratops", "worm snake", "ring-necked snake", "eastern hog-nosed snake", "smooth green snake", "kingsnake", "garter snake", "water snake", "vine snake", "night snake", "boa constrictor", "African rock python", "Indian cobra", "green mamba", "sea snake", "Saharan horned viper", "eastern diamondback rattlesnake", "sidewinder rattlesnake", "trilobite", "harvestman", "scorpion", "yellow garden spider", "barn spider", "European garden spider", "southern black widow", "tarantula", "wolf spider", "tick", "centipede", "black grouse", "ptarmigan", "ruffed grouse", "prairie grouse", "peafowl", "quail", "partridge", "african grey parrot", "macaw", "sulphur-crested cockatoo", "lorikeet", "coucal", "bee eater", "hornbill", "hummingbird", "jacamar", "toucan", "duck", "red-breasted merganser", "goose", "black swan", "tusker", "echidna", "platypus", "wallaby", "koala", "wombat", "jellyfish", "sea anemone", "brain coral", "flatworm", "nematode", "conch", "snail", "slug", "sea slug", "chiton", "chambered nautilus", "Dungeness crab", "rock crab", "fiddler crab", "red king crab", "American lobster", "spiny lobster", "crayfish", "hermit crab", "isopod", "white stork", "black stork", "spoonbill", "flamingo", "little blue heron", "great egret", "bittern bird", "crane bird", "limpkin", "common gallinule", "American coot", "bustard", "ruddy turnstone", "dunlin", "common redshank", "dowitcher", "oystercatcher", "pelican", "king penguin", "albatross", "grey whale", "killer whale", "dugong", "sea lion", "Chihuahua", "Japanese Chin", "Maltese", "Pekingese", "Shih Tzu", "King Charles Spaniel", "Papillon", "toy terrier", "Rhodesian Ridgeback", "Afghan Hound", "Basset Hound", "Beagle", "Bloodhound", "Bluetick Coonhound", "Black and Tan Coonhound", "Treeing Walker Coonhound", "English foxhound", "Redbone Coonhound", "borzoi", "Irish Wolfhound", "Italian Greyhound", "Whippet", "Ibizan Hound", "Norwegian Elkhound", "Otterhound", "Saluki", "Scottish Deerhound", "Weimaraner", "Staffordshire Bull Terrier", "American Staffordshire Terrier", "Bedlington Terrier", "Border Terrier", "Kerry Blue Terrier", "Irish Terrier", "Norfolk Terrier", "Norwich Terrier", "Yorkshire Terrier", "Wire Fox Terrier", "Lakeland Terrier", "Sealyham Terrier", "Airedale Terrier", "Cairn Terrier", "Australian Terrier", "Dandie Dinmont Terrier", "Boston Terrier", "Miniature Schnauzer", "Giant Schnauzer", "Standard Schnauzer", "Scottish Terrier", "Tibetan Terrier", "Australian Silky Terrier", "Soft-coated Wheaten Terrier", "West Highland White Terrier", "Lhasa Apso", "Flat-Coated Retriever", "Curly-coated Retriever", "Golden Retriever", "Labrador Retriever", "Chesapeake Bay Retriever", "German Shorthaired Pointer", "Vizsla", "English Setter", "Irish Setter", "Gordon Setter", "Brittany dog", "Clumber Spaniel", "English Springer Spaniel", "Welsh Springer Spaniel", "Cocker Spaniel", "Sussex Spaniel", "Irish Water Spaniel", "Kuvasz", "Schipperke", "Groenendael dog", "Malinois", "Briard", "Australian Kelpie", "Komondor", "Old English Sheepdog", "Shetland Sheepdog", "collie", "Border Collie", "Bouvier des Flandres dog", "Rottweiler", "German Shepherd Dog", "Dobermann", "Miniature Pinscher", "Greater Swiss Mountain Dog", "Bernese Mountain Dog", "Appenzeller Sennenhund", "Entlebucher Sennenhund", "Boxer", "Bullmastiff", "Tibetan Mastiff", "French Bulldog", "Great Dane", "St. Bernard", "husky", "Alaskan Malamute", "Siberian Husky", "Dalmatian", "Affenpinscher", "Basenji", "pug", "Leonberger", "Newfoundland dog", "Great Pyrenees dog", "Samoyed", "Pomeranian", "Chow Chow", "Keeshond", "brussels griffon", "Pembroke Welsh Corgi", "Cardigan Welsh Corgi", "Toy Poodle", "Miniature Poodle", "Standard Poodle", "Mexican hairless dog (xoloitzcuintli)", "grey wolf", "Alaskan tundra wolf", "red wolf or maned wolf", "coyote", "dingo", "dhole", "African wild dog", "hyena", "red fox", "kit fox", "Arctic fox", "grey fox", "tabby cat", "tiger cat", "Persian cat", "Siamese cat", "Egyptian Mau", "cougar", "lynx", "leopard", "snow leopard", "jaguar", "lion", "tiger", "cheetah", "brown bear", "American black bear", "polar bear", "sloth bear", "mongoose", "meerkat", "tiger beetle", "ladybug", "ground beetle", "longhorn beetle", "leaf beetle", "dung beetle", "rhinoceros beetle", "weevil", "fly", "bee", "ant", "grasshopper", "cricket insect", "stick insect", "cockroach", "praying mantis", "cicada", "leafhopper", "lacewing", "dragonfly", "damselfly", "red admiral butterfly", "ringlet butterfly", "monarch butterfly", "small white butterfly", "sulphur butterfly", "gossamer-winged butterfly", "starfish", "sea urchin", "sea cucumber", "cottontail rabbit", "hare", "Angora rabbit", "hamster", "porcupine", "fox squirrel", "marmot", "beaver", "guinea pig", "common sorrel horse", "zebra", "pig", "wild boar", "warthog", "hippopotamus", "ox", "water buffalo", "bison", "ram (adult male sheep)", "bighorn sheep", "Alpine ibex", "hartebeest", "impala (antelope)", "gazelle", "arabian camel", "llama", "weasel", "mink", "European polecat", "black-footed ferret", "otter", "skunk", "badger", "armadillo", "three-toed sloth", "orangutan", "gorilla", "chimpanzee", "gibbon", "siamang", "guenon", "patas monkey", "baboon", "macaque", "langur", "black-and-white colobus", "proboscis monkey", "marmoset", "white-headed capuchin", "howler monkey", "titi monkey", "Geoffroy's spider monkey", "common squirrel monkey", "ring-tailed lemur", "indri", "Asian elephant", "African bush elephant", "red panda", "giant panda", "snoek fish", "eel", "silver salmon", "rock beauty fish", "clownfish", "sturgeon", "gar fish", "lionfish", "pufferfish", "abacus", "abaya", "academic gown", "accordion", "acoustic guitar", "aircraft carrier", "airliner", "airship", "altar", "ambulance", "amphibious vehicle", "analog clock", "apiary", "apron", "trash can", "assault rifle", "backpack", "bakery", "balance beam", "balloon", "ballpoint pen", "Band-Aid", "banjo", "baluster / handrail", "barbell", "barber chair", "barbershop", "barn", "barometer", "barrel", "wheelbarrow", "baseball", "basketball", "bassinet", "bassoon", "swimming cap", "bath towel", "bathtub", "station wagon", "lighthouse", "beaker", "military hat (bearskin or shako)", "beer bottle", "beer glass", "bell tower", "baby bib", "tandem bicycle", "bikini", "ring binder", "binoculars", "birdhouse", "boathouse", "bobsleigh", "bolo tie", "poke bonnet", "bookcase", "bookstore", "bottle cap", "hunting bow", "bow tie", "brass memorial plaque", "bra", "breakwater", "breastplate", "broom", "bucket", "buckle", "bulletproof vest", "high-speed train", "butcher shop", "taxicab", "cauldron", "candle", "cannon", "canoe", "can opener", "cardigan", "car mirror", "carousel", "tool kit", "cardboard box / carton", "car wheel", "automated teller machine", "cassette", "cassette player", "castle", "catamaran", "CD player", "cello", "mobile phone", "chain", "chain-link fence", "chain mail", "chainsaw", "storage chest", "chiffonier", "bell or wind chime", "china cabinet", "Christmas stocking", "church", "movie theater", "cleaver", "cliff dwelling", "cloak", "clogs", "cocktail shaker", "coffee mug", "coffeemaker", "spiral or coil", "combination lock", "computer keyboard", "candy store", "container ship", "convertible", "corkscrew", "cornet", "cowboy boot", "cowboy hat", "cradle", "construction crane", "crash helmet", "crate", "infant bed", "Crock Pot", "croquet ball", "crutch", "cuirass", "dam", "desk", "desktop computer", "rotary dial telephone", "diaper", "digital clock", "digital watch", "dining table", "dishcloth", "dishwasher", "disc brake", "dock", "dog sled", "dome", "doormat", "drilling rig", "drum", "drumstick", "dumbbell", "Dutch oven", "electric fan", "electric guitar", "electric locomotive", "entertainment center", "envelope", "espresso machine", "face powder", "feather boa", "filing cabinet", "fireboat", "fire truck", "fire screen", "flagpole", "flute", "folding chair", "football helmet", "forklift", "fountain", "fountain pen", "four-poster bed", "freight car", "French horn", "frying pan", "fur coat", "garbage truck", "gas mask or respirator", "gas pump", "goblet", "go-kart", "golf ball", "golf cart", "gondola", "gong", "gown", "grand piano", "greenhouse", "radiator grille", "grocery store", "guillotine", "hair clip", "hair spray", "half-track", "hammer", "hamper", "hair dryer", "hand-held computer", "handkerchief", "hard disk drive", "harmonica", "harp", "combine harvester", "hatchet", "holster", "home theater", "honeycomb", "hook", "hoop skirt", "gymnastic horizontal bar", "horse-drawn vehicle", "hourglass", "iPod", "clothes iron", "carved pumpkin", "jeans", "jeep", "T-shirt", "jigsaw puzzle", "rickshaw", "joystick", "kimono", "knee pad", "knot", "lab coat", "ladle", "lampshade", "laptop computer", "lawn mower", "lens cap", "letter opener", "library", "lifeboat", "lighter", "limousine", "ocean liner", "lipstick", "slip-on shoe", "lotion", "music speaker", "loupe magnifying glass", "sawmill", "magnetic compass", "messenger bag", "mailbox", "tights", "one-piece bathing suit", "manhole cover", "maraca", "marimba", "mask", "matchstick", "maypole", "maze", "measuring cup", "medicine cabinet", "megalith", "microphone", "microwave oven", "military uniform", "milk can", "minibus", "miniskirt", "minivan", "missile", "mitten", "mixing bowl", "mobile home", "ford model t", "modem", "monastery", "monitor", "moped", "mortar and pestle", "graduation cap", "mosque", "mosquito net", "vespa", "mountain bike", "tent", "computer mouse", "mousetrap", "moving van", "muzzle", "metal nail", "neck brace", "necklace", "baby pacifier", "notebook computer", "obelisk", "oboe", "ocarina", "odometer", "oil filter", "pipe organ", "oscilloscope", "overskirt", "bullock cart", "oxygen mask", "product packet / packaging", "paddle", "paddle wheel", "padlock", "paintbrush", "pajamas", "palace", "pan flute", "paper towel", "parachute", "parallel bars", "park bench", "parking meter", "railroad car", "patio", "payphone", "pedestal", "pencil case", "pencil sharpener", "perfume", "Petri dish", "photocopier", "plectrum", "Pickelhaube", "picket fence", "pickup truck", "pier", "piggy bank", "pill bottle", "pillow", "ping-pong ball", "pinwheel", "pirate ship", "drink pitcher", "block plane", "planetarium", "plastic bag", "plate rack", "farm plow", "plunger", "Polaroid camera", "pole", "police van", "poncho", "pool table", "soda bottle", "plant pot", "potter's wheel", "power drill", "prayer rug", "printer", "prison", "missile", "projector", "hockey puck", "punching bag", "purse", "quill", "quilt", "race car", "racket", "radiator", "radio", "radio telescope", "rain barrel", "recreational vehicle", "fishing casting reel", "reflex camera", "refrigerator", "remote control", "restaurant", "revolver", "rifle", "rocking chair", "rotisserie", "eraser", "rugby ball", "ruler measuring stick", "sneaker", "safe", "safety pin", "salt shaker", "sandal", "sarong", "saxophone", "scabbard", "weighing scale", "school bus", "schooner", "scoreboard", "CRT monitor", "screw", "screwdriver", "seat belt", "sewing machine", "shield", "shoe store", "shoji screen / room divider", "shopping basket", "shopping cart", "shovel", "shower cap", "shower curtain", "ski", "balaclava ski mask", "sleeping bag", "slide rule", "sliding door", "slot machine", "snorkel", "snowmobile", "snowplow", "soap dispenser", "soccer ball", "sock", "solar thermal collector", "sombrero", "soup bowl", "keyboard space bar", "space heater", "space shuttle", "spatula", "motorboat", "spider web", "spindle", "sports car", "spotlight", "stage", "steam locomotive", "through arch bridge", "steel drum", "stethoscope", "scarf", "stone wall", "stopwatch", "stove", "strainer", "tram", "stretcher", "couch", "stupa", "submarine", "suit", "sundial", "sunglasses", "sunglasses", "sunscreen", "suspension bridge", "mop", "sweatshirt", "swim trunks / shorts", "swing", "electrical switch", "syringe", "table lamp", "tank", "tape player", "teapot", "teddy bear", "television", "tennis ball", "thatched roof", "front curtain", "thimble", "threshing machine", "throne", "tile roof", "toaster", "tobacco shop", "toilet seat", "torch", "totem pole", "tow truck", "toy store", "tractor", "semi-trailer truck", "tray", "trench coat", "tricycle", "trimaran", "tripod", "triumphal arch", "trolleybus", "trombone", "hot tub", "turnstile", "typewriter keyboard", "umbrella", "unicycle", "upright piano", "vacuum cleaner", "vase", "vaulted or arched ceiling", "velvet fabric", "vending machine", "vestment", "viaduct", "violin", "volleyball", "waffle iron", "wall clock", "wallet", "wardrobe", "military aircraft", "sink", "washing machine", "water bottle", "water jug", "water tower", "whiskey jug", "whistle", "hair wig", "window screen", "window shade", "Windsor tie", "wine bottle", "airplane wing", "wok", "wooden spoon", "wool", "split-rail fence", "shipwreck", "sailboat", "yurt", "website", "comic book", "crossword", "traffic or street sign", "traffic light", "dust jacket", "menu", "plate", "guacamole", "consomme", "hot pot", "trifle", "ice cream", "popsicle", "baguette", "bagel", "pretzel", "cheeseburger", "hot dog", "mashed potatoes", "cabbage", "broccoli", "cauliflower", "zucchini", "spaghetti squash", "acorn squash", "butternut squash", "cucumber", "artichoke", "bell pepper", "cardoon", "mushroom", "Granny Smith apple", "strawberry", "orange", "lemon", "fig", "pineapple", "banana", "jackfruit", "cherimoya (custard apple)", "pomegranate", "hay", "carbonara", "chocolate syrup", "dough", "meatloaf", "pizza", "pot pie", "burrito", "red wine", "espresso", "tea cup", "eggnog", "mountain", "bubble", "cliff", "coral reef", "geyser", "lakeshore", "promontory", "sandbar", "beach", "valley", "volcano", "baseball player", "bridegroom", "scuba diver", "rapeseed", "daisy", "yellow lady's slipper", "corn", "acorn", "rose hip", "horse chestnut seed", "coral fungus", "agaric", "gyromitra", "stinkhorn mushroom", "earth star fungus", "hen of the woods mushroom", "bolete", "corn cob", "toilet paper"]
    
    imagenet_templates = [
        'itap of a {}.',
        'a bad photo of the {}.',
        'a origami {}.',
        'a photo of the large {}.',
        'a {} in a video game.',
        'art of the {}.',
        'a photo of the small {}.',
    ]
    
    def __init__(self, clip_model, prompt_templates=imagenet_templates, device=None):
        super(PromptEnsemble, self).__init__()
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.clip_model = clip_model        
        self.templates = prompt_templates
        self.device = device
        
    def forward(self, label_idxs):
        class_names = [self.imagenet_classes[idx] for idx in label_idxs] # Convert labels to class names
        texts = [template.format(classname) for classname in class_names for template in self.templates] # Format with class names
        texts = clip.tokenize(texts).to(self.device) # Tokenize
        
        text_embeds, text_encodings = self.clip_model.embed_text(texts)
        
        text_embeds = text_embeds.reshape(len(label_idxs), len(self.templates), -1)
        text_embeds = text_embeds.mean(dim=1)
        text_embeds /= text_embeds.norm(dim=-1, keepdim=True)
             
        text_encodings = text_encodings.reshape(len(label_idxs), len(self.templates), *text_encodings.shape[-2:])
        text_encodings = text_encodings.mean(dim=1)
        text_encodings /= (text_encodings.norm(dim=-1, keepdim=True) + 1e-8)
            
        return text_embeds, text_encodings
    
class TextVisualAlignment(nn.Module):
    def __init__(self, text_dim, visual_dim, hidden_dim):
        super(TextVisualAlignment, self).__init__()
        self.text_dim = text_dim
        self.visual_dim = visual_dim
        self.hidden_dim = hidden_dim

        self.text_fc = nn.Linear(text_dim, hidden_dim)
        self.visual_fc = nn.Linear(visual_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(0.2)
        self.out = nn.Linear(hidden_dim, visual_dim)  # output dimension is visual_dim

    def forward(self, text, visual):
        text = self.text_fc(text)
        visual = self.visual_fc(visual)

        # Compute the similarity matrix between text and visual features
        # using matrix multiplication between text and visual embeddings
        sim_matrix = torch.matmul(text, visual.t())

        # Compute the attention weights for the text embeddings
        # using the softmax function along the visual dimension
        attn_weights = nn.functional.softmax(sim_matrix, dim=-1)

        # Compute the aligned text embeddings by taking the weighted sum of
        # the visual embeddings, where the weights are given by the attention
        # weights computed in the previous step
        aligned_text = torch.matmul(attn_weights.t(), text)

        # Pass the aligned text embeddings through a feedforward network
        x = self.fc(aligned_text)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.out(x)

        return x

class TextToImageDiffusion(nn.Module):
    def __init__(self, image_embed_dim=1000, text_embed_dim=512, 
                 image_embed_scale = None,
                 timesteps = 100,
                 cond_drop_prob = 0.2,
                 condition_on_text_encodings=True):
        super(TextToImageDiffusion, self).__init__()
        
        self.image_embed_dim = image_embed_dim
        self.text_embed_dim = text_embed_dim
        self.timesteps = timesteps
        
        prior_network = DiffusionPriorNetwork(
            dim = image_embed_dim,
            depth = 6,
            dim_head = 64,
            heads = 8,            
        )
        
        if image_embed_dim != text_embed_dim:
            self.project_text_embedding = nn.Linear(text_embed_dim, image_embed_dim)
        
        if image_embed_dim != text_embed_dim and condition_on_text_encodings==True:
            self.project_text_encoding = nn.Linear(text_embed_dim, image_embed_dim)
            
        self.diffusion_prior = DiffusionPrior(
            net = prior_network,
            image_embed_dim = image_embed_dim,
            timesteps = timesteps,
            cond_drop_prob = cond_drop_prob,
            condition_on_text_encodings = condition_on_text_encodings,
            image_embed_scale = image_embed_scale
        )
    
    def sample(self, text_embed=None, text_encodings=None, t=None, clip_denoised=True, device=None):
        batch_size = text_embed.shape[0]
        if device is None: device = 'cuda' if torch.cuda.is_available() else 'cpu'
        text_cond = {}
        if text_embed is not None: 
            text_embed = self.project_text_embedding(text_embed)
            text_cond = dict(**text_cond, text_embed=text_embed)
        
        if text_encodings is not None:
            text_encodings = self.project_text_encoding(text_encodings)
            text_cond = dict(**text_cond, text_encodings=text_encodings)
        
        noise = torch.rand((batch_size, self.image_embed_dim)).to(device)

        if t is None:
            t = torch.full((batch_size,), self.timesteps-1, device = device, dtype = torch.long)
                
        generated_image_embed, _ = self.diffusion_prior.p_sample(
            x = noise,
            t = t,
            text_cond = text_cond,
            clip_denoised = clip_denoised
        )
        
        return generated_image_embed
    
    def forward(self, image_embeds=None, text_embeds=None, text_encodings=None):
        
        if hasattr(self, 'project_text_embedding'):
            text_embeds = self.project_text_embedding(text_embeds)
        
        if hasattr(self, 'project_text_encoding'):
            text_encodings = self.project_text_encoding(text_encodings)
                
        loss = self.diffusion_prior(
            text_embed = text_embeds,
            text_encodings = text_encodings,
            image_embed = image_embeds,
        )
        
        return loss
    
def load_clip_aligned_model(model_name, clip_name='ViT-B/16', text_dim=768, visual_dim=768, hidden_dim=768, device=None, **kwargs):    
    
    if device is None: 
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    clip_model = OpenAIClipAdapter(name=clip_name)
    input_resolution = clip_model.clip.visual.input_resolution
    context_length = clip_model.clip.context_length
    vocab_size = clip_model.clip.vocab_size

    print("Clip Model parameters:", f"{np.sum([int(np.prod(p.shape)) for p in clip_model.parameters()]):,}")
    print("Input resolution:", input_resolution)
    print("Context length:", context_length)
    print("Vocab size:", vocab_size)
    
    label_encoder = PromptEnsemble(clip_model=clip_model)
    print("label_encoder templates")
    print(label_encoder.templates)
    
    # model = TextVisualAlignment(text_dim=768, visual_dim=1000, hidden_dim=768).to(device)
    model = TextToImageDiffusion()
    model.to(device)
    
    weights_url = weight_urls[model_name]
    checkpoint = torch.hub.load_state_dict_from_url(weights_url, 
                                                    map_location='cpu', 
                                                    check_hash=True,
                                                    **kwargs)
    msg = model.load_state_dict(checkpoint['state_dict'])
    print(msg)
    
    return label_encoder, model
