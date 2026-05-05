# Curated Test Set - Image Generation Prompts

Prompts for generating 22 evaluation images with Gemini Nano Banana.

Decision rule used to assign expected actions:
- CONTINUE: path is clear, no obstacle in the operation zone
- STOP: a moving obstacle (person, animal, child) is in or about to enter the path
- REROUTE: a stationary obstacle (object, furniture, plant, sitting/standing person) is in the path

Camera viewpoint for every image: low to the ground (around 20 to 30 cm above grass level), looking forward across a residential garden lawn, simulating the perspective of a robotic lawn mower. The mower body itself should NOT be visible in the frame.

Conditions are intentionally varied. Not every scene is sunny, not every lawn is pristine.

---

## CONTINUE scenarios (4 images)

### C01 - empty mowed lawn, midday
Photorealistic outdoor photograph from a low camera angle around 25 cm above the ground, looking forward across a freshly mowed suburban back lawn. The grass extends ahead for several metres before reaching a low wooden fence and the corner of a neighbouring house in the background. Bright midday light, clear blue sky with a few wispy clouds. No people, no animals, no objects on the grass. Slight visible mower stripes in the lawn. Sharp focus, natural colour grading.

### C02 - empty lawn after rain, overcast
Photorealistic outdoor photograph from a low camera angle near grass level, looking across a slightly damp residential lawn after recent rain. Grey overcast sky, soft diffuse light, no sun. Visible water droplets on a few blades of grass in the foreground. Lawn extends to a tall hedge in the background. Lawn is healthy but not perfectly trimmed, a few small clover patches and dandelion leaves visible. No people, no animals, no objects.

### C03 - distant person well outside operation zone
Photorealistic outdoor photograph from a low ground-level camera angle, looking across a residential garden lawn during late afternoon golden-hour light. Clear path ahead with neatly mowed grass. In the far background, well past a low hedge that marks the end of the garden, a person can be seen walking along a public footpath on the other side of the hedge. The person appears small and is clearly outside the lawn area. Long warm shadows on the grass. Soft side-lighting.

### C04 - empty lawn at dusk
Photorealistic outdoor photograph from a low ground-level camera angle, looking across an empty suburban back lawn at early dusk. The sky is a deep blue-purple gradient with the last warm light just visible at the horizon behind a row of trees. Long shadows stretch across the grass. The lawn is well kept but not pristine, with a few visible bare patches near the edges. No obstacles, no people, no animals. Slightly cool, blueish ambient light.

---

## STOP scenarios (8 images, all moving obstacles)

### S01 - adult walking directly toward the mower
Photorealistic outdoor photograph from a low ground-level camera angle on a residential back lawn, harsh midday sun overhead. An adult man in jeans and a plain grey t-shirt is walking directly toward the camera, currently about three metres ahead, centred in the frame, mid-stride with one foot forward and the other behind. Casual relaxed posture, arms swinging. Slight motion blur on the trailing leg. Bright clear sky, sharp shadows on the grass. The lawn is well kept.

### S02 - adult walking laterally across the path
Photorealistic outdoor photograph from a low camera angle near grass level, looking across a back garden lawn under partly cloudy skies, mild and breezy. An adult woman in casual outdoor clothes, jeans and a light jumper, is walking across the frame from left to right, currently about two and a half metres ahead, mid-stride. She is looking ahead, not at the camera. The grass is slightly long with a few weeds and dandelions visible. Wooden fence in the background. Soft alternating light and shade as clouds move overhead.

### S03 - jogger crossing diagonally
Photorealistic outdoor photograph from a low ground-level camera angle on a residential lawn in the cool early morning light shortly after sunrise. An adult in running gear, shorts and a fitted top, is jogging diagonally across the operation zone about three metres ahead, moving from the right side toward the upper left. Slight motion blur in the legs and arms. Damp grass with visible dewdrops catching the light. Soft warm sunlight from the side casting long shadows.

### S04 - child running across grass
Photorealistic outdoor photograph from a low camera angle close to the lawn, bright afternoon light. A young child of around five years old in a striped t-shirt and shorts is running across the grass about two metres ahead, hair lifted by the motion, one foot in the air. The child is looking off to the side, mid-laugh. A garden swing and a ball visible in the far background near a fence. Lawn is well-used with a few worn patches.

### S05 - dog running across lawn
Photorealistic outdoor photograph from a low ground-level camera angle on a suburban lawn under slightly overcast skies. A medium-sized brown labrador-type dog is mid-stride running across the frame from the right side, about two metres ahead, ears flapping, tongue out, all four paws off the ground. The dog is clearly in motion. Lawn is slightly uneven with patchy grass and a few bare spots. A garden shed visible in the background. Even, diffused light.

### S06 - cat walking through danger zone
Photorealistic outdoor photograph from a low camera angle in late morning light with dappled shade from an unseen tree. A grey domestic shorthair cat is walking through the grass about one and a half metres ahead, mid-step with one front paw raised, head slightly turned toward something off-camera to the right. The lawn is slightly long, in need of mowing, with visible clover. Realistic lived-in suburban back garden. Mottled light and shadow on the grass.

### S07 - adult bending to pick something up
Photorealistic outdoor photograph from a low ground-level camera angle under soft cloudy daylight. An adult woman is bent over at the waist about two metres ahead in the centre of the frame, picking something off the grass with her right hand, gardening clothes and a wide-brimmed hat, hair partly tied back. She is in active motion, mid-reach. The lawn has visible weeds and a few fallen leaves. Soft even light, no harsh shadows.

### S08 - two adults walking across operation zone
Photorealistic outdoor photograph from a low camera angle close to the grass, warm afternoon light. Two adults, one in a summer dress and one in shorts and a t-shirt, are walking side by side from left to right across the lawn, currently about three metres ahead, mid-stride, talking to each other, mouths slightly open. Both are clearly in motion. Slightly overgrown lawn with visible weeds, residential garden setting with a wooden fence and a glimpse of a garden shed in the background.

---

## REROUTE scenarios (10 images, all stationary obstacles)

### R01 - empty wooden garden chair
Photorealistic outdoor photograph from a low ground-level camera angle on a back garden lawn in bright daylight. A weathered wooden Adirondack chair sits empty on the grass about two metres ahead of the camera, slightly angled toward the right. The chair shows clear signs of age: faded blue paint, a small water stain on the seat slats, slightly worn armrests. Lawn is well-kept but realistic, with a few small clover patches. Clear sky, sharp shadow under the chair on the grass.

### R02 - plastic outdoor table
Photorealistic outdoor photograph from a low camera angle near grass level under overcast skies. A round white plastic outdoor table is placed in the middle of the lawn about two and a half metres ahead. The table is empty, slightly weathered with visible grey staining and minor scuffs on the surface. Soft diffuse light, no harsh shadows. The lawn surrounding the table is slightly flattened where the table base has been sitting for a while. A couple of fallen leaves on the table.

### R03 - raised flower bed in operation zone
Photorealistic outdoor photograph from a low ground-level camera angle in early morning sunlight with slight dew on the grass. A small raised flower bed bordered with weathered wooden planks sits in the operation zone about three metres ahead, planted with low colourful flowers, marigolds and pansies, and a few leafy plants. The bed protrudes about 15 to 20 cm above the surrounding lawn. Realistic suburban garden, lawn around the bed shows some bare patches and worn earth near the wood edges.

### R04 - garden rake on grass
Photorealistic outdoor photograph from a low camera angle close to the grass under a cloudy sky. A metal-tined garden rake lies flat across the lawn about two metres ahead, with the wooden handle pointing toward the upper left of the frame and the head toward the lower right. Some scattered fallen leaves are visible near the rake head. The grass is slightly long. Even, soft light without harsh shadows. Realistic suburban back garden setting.

### R05 - football on lawn
Photorealistic outdoor photograph from a low ground-level camera angle in late afternoon sunlight, warm side-lighting. A worn black-and-white football is sitting still on the lawn about one and a half metres ahead, slightly off-centre to the right. The ball shows visible scuffs, dirt marks and a partially deflated patch. The grass is well-kept but has a few patches of dirt where it has been played on. Realistic back garden with a wooden fence in the background and a hint of a child's bicycle leaning against the fence.

### R06 - coiled garden hose
Photorealistic outdoor photograph from a low camera angle close to the grass, in shaded indirect light under a tree canopy. A green coiled garden hose lies partly tangled on the lawn about two metres ahead, with one loop trailing toward the camera. The hose is slightly damp and dirty. Lawn has uneven mowing and visible imperfections, a few weeds. A garden tap fixture is just visible at the edge of the frame on the right.

### R07 - abandoned children's scooter
Photorealistic outdoor photograph from a low ground-level camera angle in bright open daylight under a clear sky. A child's pink and silver kick scooter lies on its side on the lawn about two metres ahead, handlebar pointing toward the camera. The scooter is well-used with visible scratches and stickers. The lawn around it has slight wear marks and a few bare patches from regular play. Realistic suburban back garden, fence visible in the background.

### R08 - person sitting on grass reading a book
Photorealistic outdoor photograph from a low camera angle near grass level, warm afternoon sunlight. An adult woman sits cross-legged on the lawn about three metres ahead in the centre of the frame, holding a paperback book in both hands and reading. Calm settled posture, eyes on the page, no movement. Casual summer clothes, light cardigan, hair loose. The grass around her is well-kept but not perfect, with a few clover patches. Soft shadows from a tree off-frame.

### R09 - person sunbathing on towel
Photorealistic outdoor photograph from a low ground-level camera angle in clear midday sunlight. An adult is lying flat on a striped beach towel on the lawn about three metres ahead, sunglasses on, arms relaxed by their sides, headphones in, clearly resting and motionless. The towel covers most of their body. Bright sun, sharp shadows from a nearby fence post just visible at the side of the frame. The grass is dry and slightly yellowed in patches from the sun.

### R10 - person standing still on phone
Photorealistic outdoor photograph from a low camera angle close to the lawn under overcast skies, soft even light. An adult man is standing still in the centre of the frame about two and a half metres ahead, looking down at his mobile phone, both feet planted, weight slightly on his back foot. Stationary posture, no motion blur, no walking stance. Casual clothes, dark jeans and a hoodie, hands holding the phone in front of him. Realistic suburban garden, slightly overgrown lawn with visible weeds and a fence in the background.

---

## Naming convention for the generated files

Save each generated image into `evaluation/curated_test/images/` using the scenario ID as the filename, e.g. `C01.jpg`, `S05.jpg`, `R10.jpg`. The script will read these IDs against `ground_truth.csv` to score predictions.

Adding more scenarios later only requires:
1. Drop the new image into `images/` with a new ID, e.g. `R11.jpg`.
2. Add a corresponding row to `ground_truth.csv`.

No code changes needed.
