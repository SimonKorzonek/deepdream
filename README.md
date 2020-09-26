- D E E P D R E M  R E A D M E -


URL ----------> Path of file to dreamify


chosen_layers --> Layers of dream model selected to attempt in object recognition
                process. Lower layers will see most basic patterns. Higher ones
                will produce more complex patterns made from sum of simpler patterns.
                Named from 'mixed0' to 'mixed9'.
                - YOU HAVE TO CHOSE AT LEAST 2 LAYERS.
                - HIGHER LAYERS ARE COMPUTIONALY HEAVIER AND MORE RAM CONSUMING


OCTAVES ------> Amount of iterations of gradient ascent applied on different scales.
                This allowes patterns generated at smaller scales to be incorporated
                into patterns at higher scales and filled with additional detail.
                Octaves allows to generate patterns within patterns
                - Reccomended values: 5 - 20


OCTAVE_SCALE -> Ratio of image upscaling at each octave iteration.
                10 octaves with 1.3 scaling gives dream in 4k and requires 12GB RAM


STEP_SIZE ----> Higher value amplifies dreamyfied patterns intensivity,
                but this doesn't always mean that's gonna look better. Higher
                values work well with greater number of used dream layers.
                Complex-color source images are works better with smaller steps,
                they're not messing colors that much
                - Recommanded values: 0.05 - 0.2


TARGET_SIZE --> Downsizing image will give abstraction on higher level
                of source_image composition - you can increase OCTAVE_SCALE
                to make autput bigger. Upsizing image will produce more
                detailed chaos, decrease OCTAVE_SCALE
                You have to respect proportions in original image resolution,
                so dreamified image doesn't stretch.
                    If source_photo is in x:y aspect ratio and x > y:
                        if vertical: x:y
                        if horizontal: y:x
                Higher values of target size will cause dream to attach more
                dreamified patterns to source image detail, but in InceptionV3
                model these patterns will get smaller than in case of smaller value
                Smaller target size will blur source_image,
                but it will get processed faster
