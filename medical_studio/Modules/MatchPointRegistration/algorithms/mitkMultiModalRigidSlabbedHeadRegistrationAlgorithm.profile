SET(ALGORITHM_PROFILE_UID_Namespace "org.mitk")
SET(ALGORITHM_PROFILE_UID_Name "MultiModal.rigid.slabbedHead")
SET(ALGORITHM_PROFILE_UID_Version "1.0.0")

SET(ALGORITHM_PROFILE_Description "Algorithm is used to solve a special case of head registration problems. It is configured to register the slabbed MRI head data (thus data where only a part of the head is visible) onto whole Head images (e.g.CT planning data). Differenz to the default rigid algorithm is that this algorithms is very defensive with rotating out of the slice plan (so roll or pitch). Uses 3 Resolution levels. By default initializes via image centers.")
SET(ALGORITHM_PROFILE_Contact "Ralf Floca\; mitk-users@lists.sourceforge.net")

SET(ALGORITHM_PROFILE_DataType "Image")
SET(ALGORITHM_PROFILE_ResolutionStyle "3 (multi res)")
SET(ALGORITHM_PROFILE_DimMoving "3")
SET(ALGORITHM_PROFILE_ModalityMoving "MR" "any")
SET(ALGORITHM_PROFILE_DimTarget "3")
SET(ALGORITHM_PROFILE_ModalityTarget "MR" "any")
SET(ALGORITHM_PROFILE_Subject "any")
SET(ALGORITHM_PROFILE_Object "Head")
SET(ALGORITHM_PROFILE_TransformModel "rigid")
SET(ALGORITHM_PROFILE_Metric "Mattes mutual information")
SET(ALGORITHM_PROFILE_TransformDomain "global")
SET(ALGORITHM_PROFILE_Optimization "Regular Step Gradient Descent")
SET(ALGORITHM_PROFILE_Keywords "basic" "pre initialization" "Head" "slabbed" "partial" "multimodal" "rigid")