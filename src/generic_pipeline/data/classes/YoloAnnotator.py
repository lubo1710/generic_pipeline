from dataclasses import dataclass
import robokudo

liste = [
    'Fork',
    'Pitcher',
    'Bleachcleanserbottle',
    'Crackerbox',
    'Minisoccerball',
    'Baseball',
    'Mustardbottle',
    'Jellochocolatepuddingbox',
    'Wineglass',
    'Orange',
    'Coffeepack',
    'Softball',
    'Metalplate',
    'Pringleschipscan',
    'Strawberry',
    'Glasscleanerspraybottle',
    'Tennisball',
    'Spoon',
    'Metalmug',
    'Abrasivesponge',
    'Jellobox',
    'Dishwashertab',
    'Knife',
    'Cerealbox',
    'Metalbowl',
    'Sugarbox',
    'Coffeecan',
    'Milkpackja',
    'Apple',
    'Tomatosoupcan',
    'Tunafishcan',
    'Gelatinebox',
    'Pear',
    'Lemon',
    'Banana',
    'Pottedmeatcan',
    'Peach',
    'Plum',
    'Rubikscube',
    'Mueslibox',
    'Cupblue',
    'Cupgreen',
    'Largemarker',
    'Masterchefcan',
    'Scissors',
    'Scrubcleaner',
    'Grapes',
    'Cup_small',
    'screwdriver',
    'clamp',
    'hammer',
    'wooden_block',
    'Cornybox',
    'object']

@dataclass()
class Annotator:
    name = 'YoloAnnotator'
    source =  'generic_pipeline.annotators.YoloAnnotator'
    description = 'Detects objects in images and creates a ObjectHypothesis'
    descriptor = {}
    parameters = {
    'precision_mode' : True,
    'ros_pkg_path' : 'generic_pipeline',
    'weights_path' : 'src/generic_pipeline/Weights-SUTURO23/data_03_24.pt',
    'id2name_json_path' : 'src/generic_pipeline/data/json/id2name_edit.json',
    'threshold' : 0.6
    }
    inputs = []
    outputs = [robokudo.types.scene.ObjectHypothesis , robokudo.types.annotation.Classification, robokudo.types.annotation.Shape]
    capabilities = {robokudo.types.scene.ObjectHypothesis : liste,
                    robokudo.types.annotation.Classification : liste}