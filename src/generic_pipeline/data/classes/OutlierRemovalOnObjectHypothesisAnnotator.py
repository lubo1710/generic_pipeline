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
    'Milk',
    'Apple',
    'Tomatosoupcan',
    'Tunafishcan',
    'Gelatinebox',
    'Pear',
    'Lemon',
    'Banana',
    'Meatcan',
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
    'Cupsmall',
    'Screwdriver',
    'Clamp',
    'Hammer',
    'Woodenblock',
    'Cornybox',
    'object',
    'cup',
    'muesli',
    'fruit',
    'dish',
    'cutlery',
    'tool',
    'toy',
    'ball',
    'food',
    'drink',
    'coffee',
    'cleaning_tool']

@dataclass()
class Annotator:
    name = 'OutlierRemovalOnObjectHypothesisAnnotator'
    source =  'generic_pipeline.annotators.outlier_removal_objecthypothesis'
    description = 'Removes outlier from OH'
    descriptor = {}
    parameters = {}
    inputs = [robokudo.types.annotation.StampedPoseAnnotation, robokudo.cas.CASViews.CLOUD]
    outputs = [robokudo.types.scene.ObjectHypothesis , robokudo.types.annotation.Classification, robokudo.types.annotation.Shape]
    capabilities = {robokudo.types.scene.ObjectHypothesis : liste,
                    robokudo.types.annotation.Classification : liste}