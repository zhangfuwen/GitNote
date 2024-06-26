
```plantuml
@startuml

component VisionEmbedding {
	portin image
	component patch_embedding
	component position_embedding
	component class_embedding
	portout features
}

input --> image



@enduml
```