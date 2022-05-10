import { BrowserModule } from '@angular/platform-browser';
import { NgModule } from '@angular/core';
import { AppRoutingModule } from './app-routing.module';
import { AppComponent } from './app.component';
import { BrowserAnimationsModule } from '@angular/platform-browser/animations';
import { MatCardModule} from '@angular/material/card';
import { MatToolbarModule } from '@angular/material/toolbar';
import { MatButtonModule } from '@angular/material/button';
import { FlexLayoutModule } from '@angular/flex-layout';
import { TagGeneratorComponent } from './tag-generator/tag-generator.component';
import { CaptionGeneratorComponent } from './caption-generator/caption-generator.component';
import { HashtagListComponent } from './hashtag-list/hashtag-list.component';
import {MatGridListModule } from '@angular/material/grid-list';
import {HttpClientModule} from '@angular/common/http';
import { TagsDisplayComponent } from './tags-display/tags-display.component';
import { CaptionDisplayComponent } from './caption-display/caption-display.component';
import { FriendsListComponent } from './friends-list/friends-list.component'


@NgModule({
  declarations: [
    AppComponent,
    TagGeneratorComponent,
    CaptionGeneratorComponent,
    HashtagListComponent,
    TagsDisplayComponent,
    CaptionDisplayComponent,
    FriendsListComponent,
   
  ],
  imports: [
    BrowserModule,
    AppRoutingModule,
    BrowserAnimationsModule,
    MatGridListModule,
    MatCardModule,
    MatToolbarModule,
    MatButtonModule,
    FlexLayoutModule,
    HttpClientModule,
   
    
    
  ],
  providers: [],
  bootstrap: [AppComponent]
})
export class AppModule { }
