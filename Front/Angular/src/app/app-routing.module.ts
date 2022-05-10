import { NgModule } from '@angular/core';
import { RouterModule, Routes } from '@angular/router';
import { CaptionDisplayComponent } from './caption-display/caption-display.component';
import { CaptionGeneratorComponent } from './caption-generator/caption-generator.component';
import { FriendsListComponent } from './friends-list/friends-list.component';
import { HashtagListComponent } from './hashtag-list/hashtag-list.component';
import { TagGeneratorComponent } from './tag-generator/tag-generator.component';
import { TagsDisplayComponent } from './tags-display/tags-display.component';

const routes: Routes = [
  { path: 'app-tag-generator', component: TagGeneratorComponent },
  { path: 'app-caption-generator', component: CaptionGeneratorComponent},
  { path: 'app-hashtag-list', component: HashtagListComponent},
  { path: 'app-tags-display', component:TagsDisplayComponent},
  { path: 'app-caption-display', component:CaptionDisplayComponent},
  { path : 'app-friends-list', component:FriendsListComponent},
];

@NgModule({
  imports: [RouterModule.forRoot(routes)],
  exports: [RouterModule]
})
export class AppRoutingModule { }
