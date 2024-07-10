from dataclasses import dataclass
import robokudo

liste = [
    'fork',
    'pitcher',
    'bleach_cleanser_bottle',
    'cracker_box',
    'mini_soccer_ball',
    'baseball',
    'mustard_bottle',
    'jello_chocolate_pudding_box',
    'wineglass',
    'orange',
    'coffee_pack',
    'softball',
    'metal_plate',
    'pringles_chips_can',
    'strawberry',
    'glass_cleaner_spray_bottle',
    'tennis_ball',
    'spoon',
    'metal_mug',
    'abrasive_sponge',
    'jello_box',
    'dishwasher_tab',
    'knife',
    'cerealbox',
    'metal_bowl',
    'sugar_box',
    'coffee_can',
    'milk',
    'apple',
    'tomato_soupcan',
    'tuna_fish_can',
    'gelatine_box',
    'pear',
    'lemon',
    'banana',
    'meat_can',
    'peach',
    'plum',
    'rubikscube',
    'muesli_box',
    'cup_blue',
    'cup_green',
    'large_marker',
    'master_chef_can',
    'scissors',
    'scrub_cleaner',
    'grapes',
    'cup_small',
    'screwdriver',
    'clamp',
    'hammer',
    'wooden_block',
    'corny_box',
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